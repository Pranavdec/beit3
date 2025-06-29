import json
import random
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import XLMRobertaTokenizer

import spectral.vqa_data as vqa_data
from perturbation_helper import get_image_relevance, get_text_relevance
import modeling_finetune  
from timm.models import create_model
import utils

VQA_URL = 'vqa_dict.json'


def load_answer_dict():
    with open(VQA_URL, 'r') as fp:
        return json.load(fp)


class AttentionHook:
    def __init__(self, module):
        self.attn = None
        self.grad = None
        # 1) on forward—store the raw attn‐probs (out[1])
        module.register_forward_hook(self._forward_hook)
        # 2) on backward—grab grad_out[1]
        module.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        # out is (attn_output, attn_probs)
        if isinstance(out, tuple) and out[1] is not None:
            self.attn = out[1]

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out is a tuple (dL/d out[0], dL/d out[1])
        if isinstance(grad_out, tuple) and len(grad_out) > 1:
            self.grad = grad_out[1]





class BEIT3Perturber:
    def __init__(self, ckpt_path: str, tokenizer_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
        model_config = "beit3_base_patch16_480_vqav2"
        self.model = create_model(
            model_config,
            pretrained=False,
            drop_path_rate=0.1,
            vocab_size=64010,
            checkpoint_activations=False,
        )

        utils.load_model_and_may_interpolate(ckpt_path, self.model, 'model', '')

        self.model.to(self.device)
        self.model.eval()

        self.answer_dict = load_answer_dict()
        # self.hooks = [AttentionHook(layer.self_attn) for layer in self.model.beit3.encoder.layers]

        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def encode(self, image_path: str, question: str):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        tokens = self.tokenizer(question, return_tensors='pt')
        text_ids = tokens['input_ids'].to(self.device)
        masks = tokens['attention_mask'].to(self.device)
        return image, text_ids, masks

    def forward(self, image, question_ids, masks):
        outputs = self.model.beit3(textual_tokens=question_ids,
                                   visual_tokens=image,
                                   text_padding_position=masks)
        feats = outputs['encoder_out']
        cls = self.model.pooler(feats)
        logits = self.model.head(cls)
        return logits, feats

    def predict(self, image_path: str, question: str):
        img, txt, mask = self.encode(image_path, question)
        self.model.zero_grad()
        logits, feats = self.forward(img, txt, mask)
        # backpropagate on the predicted answer to get attention gradients
        pred_idx = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).to(self.device)
        one_hot[0, pred_idx] = 1.0
        loss = (logits * one_hot).sum()
        loss.backward(retain_graph=True)
        answer = self.answer_dict.get(str(pred_idx.item()), '')
        img_len = self.model.beit3.vision_embed.num_position_embeddings()
        return answer, feats.detach(), img_len

    def relevance_upse(self, feats, img_len):
        grads = [h.self_attn.get_attn_gradients() for h in self.model.beit3.encoder.layers]
        cams = [h.self_attn.get_attention_map() for h in self.model.beit3.encoder.layers]
        ret = {
            'image_feats': feats[:, 1:img_len, :],
            'text_feats': feats[:, img_len:, :]
        }
        print(f"Number of layers: {len(grads)}")
        for i in range(len(grads)):
            print(f"Layer {i} - Grad shape: {grads[i].shape}, CAM shape: {cams[i].shape}")
        # print shapes of image and text features
        print(f"Image feats shape: {ret['image_feats'].shape}, Text feats shape: {ret['text_feats'].shape}")
        
        t_rel = get_text_relevance(ret, grads, cams)
        i_rel = get_image_relevance(ret, grads, cams)
        return torch.tensor(t_rel), torch.tensor(i_rel)

    def relevance_rm(self, feats, img_len):
        # reuse UPSE helper with gradients from attention
        return self.relevance_upse(feats, img_len)


class PerturbationRunner:
    def __init__(self, perturber: BEIT3Perturber):
        self.perturber = perturber
        self.dataset = vqa_data.VQADataset(splits='valid')
        self.steps = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1]
        self.acc = [0.0] * len(self.steps)

    def perturb(self, item, img_rel, txt_rel):
        image_file = item['img_id'] + '.jpg'
        _, feats, img_len = self.perturber.predict(image_file, item['sent'])
        text_embeds = feats[:, img_len:, :]
        text_mask = torch.ones(text_embeds.shape[1], dtype=torch.long).unsqueeze(0).to(self.perturber.device)
        image_embeds = feats[:, :img_len, :]

        for i, step in enumerate(self.steps):
            cam_pure = txt_rel[1:-1]
            keep_t = int((1 - step) * cam_pure.shape[0])
            _, idx = txt_rel[1:-1].topk(k=keep_t)
            keep_idx = [0, txt_rel.shape[0]-1] + [j+1 for j in idx.cpu().numpy()]
            keep_idx = sorted(keep_idx)
            curr_text = text_embeds[:, keep_idx, :]
            curr_mask = text_mask[:, keep_idx]
            logits, _ = self.perturber.forward(image_embeds, curr_text, curr_mask)
            pred = self.perturber.answer_dict.get(str(logits.argmax().item()), '')
            acc = item['label'].get(pred, 0)
            self.acc[i] += acc
        return self.acc

    def run(self, num_samples=5, method='rm', images_folder=None):
        idxs = list(range(len(self.dataset.data)))
        random.shuffle(idxs)
        idxs = idxs[:num_samples]
        for idx in idxs:
            item = self.dataset.data[idx]
            img_path = images_folder + item['img_id'] + '.jpg'
            _, feats, img_len = self.perturber.predict(img_path, item['sent'])
            if method == 'rm':
                t_rel, i_rel = self.perturber.relevance_rm(feats, img_len)
            else:
                t_rel, i_rel = self.perturber.relevance_upse(feats, img_len)
            self.perturb(item, i_rel, t_rel)
        return [a / num_samples for a in self.acc]


def main():
    # Example usage placeholder
    ckpt = 'beit3_base_indomain_patch16_224.pth'
    tokenizer_path = 'beit3.spm'
    images_folder = '/home/pranav/v2_ExplanableAI/beit3/data/val2014/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pert = BEIT3Perturber(ckpt, tokenizer_path, device)
    runner = PerturbationRunner(pert)
    results = runner.run(num_samples=2, method='rm', images_folder=images_folder)
    print(results)


if __name__ == '__main__':
    main()
