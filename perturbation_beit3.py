import json
import random

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import XLMRobertaTokenizer
from tqdm import tqdm

import spectral.vqa_data as vqa_data
from perturbation_helper import get_image_relevance, get_text_relevance
import modeling_finetune 
from modeling_finetune import beit3_base_patch16_480_vqav2
from timm.models import create_model
import utils

ANSWER2LABEL = 'data/answer2label.txt'


def load_answer_dict(path: str = ANSWER2LABEL):
    """Load answer dictionary mapping index to answer string.

    The file is expected to contain one JSON object per line with
    ``{"answer": str, "label": int}`` as used during BEiT-3 fine-tuning.
    """
    answer_dict = {}
    with open(path, 'r') as fp:
        for line in fp:
            item = json.loads(line)
            answer_dict[str(item["label"])] = item["answer"]
    return answer_dict


class BEIT3Perturber:
    def __init__(self, ckpt_path: str, tokenizer_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.tokenizer = XLMRobertaTokenizer(tokenizer_path)
        model_config = "beit3_base_patch16_480_vqav2"
        self.model = beit3_base_patch16_480_vqav2(pretrained=False,
                                                  drop_path_rate=0.1,
                                                  vocab_size=64010,
                                                  checkpoint_activations=False)

        utils.load_model_and_may_interpolate(ckpt_path, self.model, 'model', '')

        self.model.to(self.device)
        self.model.eval()

        self.answer_dict = load_answer_dict()

        self.transform = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def encode(self, image_path: str, question: str, max_len: int = 64):
        """Convert raw inputs to model tensors following training preprocessing."""
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)

        tokens = self.tokenizer.tokenize(question)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(token_ids) > max_len - 2:
            token_ids = token_ids[: max_len - 2]
        token_ids = [self.tokenizer.bos_token_id] + token_ids + [self.tokenizer.eos_token_id]
        pad_len = max_len - len(token_ids)
        text_ids = token_ids + [self.tokenizer.pad_token_id] * pad_len
        padding_mask = [0] * len(token_ids) + [1] * pad_len

        text_ids = torch.tensor(text_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        masks = torch.tensor(padding_mask, dtype=torch.long).unsqueeze(0).to(self.device)
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
        
        t_rel = get_text_relevance(ret, grads, cams)
        i_rel = get_image_relevance(ret, grads, cams)
        return torch.tensor(t_rel), torch.tensor(i_rel)

    def relevance_rm(self, feats, img_len):
        # reuse UPSE helper with gradients from attention
        return self.relevance_upse(feats, img_len)


class PerturbationRunner:
    def __init__(self, perturber: BEIT3Perturber, image_folder: str = None):
        self.perturber = perturber
        self.dataset = vqa_data.VQADataset(splits='valid')
        self.steps = [0, 0.25, 0.5, 0.75, 0.9, 0.95]
        self.acc_text_pos = [0.0] * len(self.steps)
        self.acc_text_neg = [0.0] * len(self.steps)
        self.acc_img_pos = [0.0] * len(self.steps)
        self.acc_img_neg = [0.0] * len(self.steps)
        self.image_folder = image_folder

    def _perturb_text(self, item, txt_rel, positive=True):
        image_file = self.image_folder + (item['img_id'] + '.jpg')
        img, txt_ids, _ = self.perturber.encode(image_file, item['sent'])

        for i, step in enumerate(self.steps):
            cam_pure = txt_rel[1:-1]
            keep_t = int((1 - step) * cam_pure.shape[0])
            if positive:
                _, idx = cam_pure.topk(k=keep_t)
            else:
                _, idx = (-cam_pure).topk(k=keep_t)
            keep_idx = [0, txt_rel.shape[0] - 1] + [j + 1 for j in idx.cpu().numpy()]
            keep_idx = sorted(keep_idx)
            curr_text = txt_ids[:, keep_idx]
            curr_mask = torch.zeros_like(curr_text)
            logits, _ = self.perturber.forward(img, curr_text, curr_mask)
            pred = self.perturber.answer_dict.get(str(logits.argmax().item()), '')
            print(f"Prediction: {pred} | Label: {item['label']}")
            acc = item['label'].get(pred, 0)
            if positive:
                self.acc_text_pos[i] += acc
            else:
                self.acc_text_neg[i] += acc

    def _apply_patch_mask(self, image, mask_vec):
        num_patches = int(mask_vec.numel())
        side = int(num_patches ** 0.5)
        patch_h = image.shape[-2] // side
        patch_w = image.shape[-1] // side
        mask_2d = mask_vec.view(side, side)
        img = image.clone()
        for r in range(side):
            for c in range(side):
                if mask_2d[r, c] == 0:
                    img[:, :, r * patch_h:(r + 1) * patch_h, c * patch_w:(c + 1) * patch_w] = 0
        return img

    def _perturb_image(self, item, img_rel, positive=True):
        image_file = self.image_folder + (item['img_id'] + '.jpg')
        img, txt_ids, mask = self.perturber.encode(image_file, item['sent'])
        img_rel = img_rel.flatten()
        for i, step in enumerate(self.steps):
            keep_p = int((1 - step) * img_rel.shape[0])
            if positive:
                _, idx = img_rel.topk(k=keep_p)
            else:
                _, idx = (-img_rel).topk(k=keep_p)
            mask_vec = torch.zeros_like(img_rel, dtype=torch.long)
            mask_vec[idx] = 1
            masked_img = self._apply_patch_mask(img, mask_vec)
            logits, _ = self.perturber.forward(masked_img, txt_ids, mask)
            pred = self.perturber.answer_dict.get(str(logits.argmax().item()), '')
            acc = item['label'].get(pred, 0)
            if positive:
                self.acc_img_pos[i] += acc
            else:
                self.acc_img_neg[i] += acc

    def run(self, num_samples=5, method='rm', images_folder=None, seed=42   ):
        random.seed(seed)
        idxs = list(range(len(self.dataset.data)))
        random.shuffle(idxs)
        idxs = idxs[:num_samples]
        for idx in tqdm(idxs, desc="samples"):
            item = self.dataset.data[idx]
            print(item)
            img_path = images_folder + item['img_id'] + '.jpg'
            _, feats, img_len = self.perturber.predict(img_path, item['sent'])
            if method == 'rm':
                t_rel, i_rel = self.perturber.relevance_rm(feats, img_len)
            else:
                t_rel, i_rel = self.perturber.relevance_upse(feats, img_len)

            self._perturb_text(item, t_rel, positive=True)
            self._perturb_text(item, t_rel, positive=False)
            self._perturb_image(item, i_rel, positive=True)
            self._perturb_image(item, i_rel, positive=False)

        results = {
            'text_positive': [a / num_samples for a in self.acc_text_pos],
            'text_negative': [a / num_samples for a in self.acc_text_neg],
            'image_positive': [a / num_samples for a in self.acc_img_pos],
            'image_negative': [a / num_samples for a in self.acc_img_neg],
        }

        for key, values in results.items():
            auc = np.trapezoid(values, self.steps)
            print(f"{key} AUC: {auc:.4f} | y values: {values}")

        return results


def main():
    # Example usage placeholder
    ckpt = 'beit3_base_indomain_patch16_224.pth'
    tokenizer_path = 'beit3.spm'
    images_folder = '/home/pranav/v2_ExplanableAI/beit3/data/val2014/'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pert = BEIT3Perturber(ckpt, tokenizer_path, device)
    runner = PerturbationRunner(pert, image_folder=images_folder)
    results = runner.run(num_samples=20, method='rm', images_folder=images_folder)
    print(results)


if __name__ == '__main__':
    main()
