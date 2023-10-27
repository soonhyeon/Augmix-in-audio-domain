import torch 
import torch.nn as nn 
import torch.nn.functional as F
import augmentation
import numpy as np 


def apply_op(audio, sr, op):
    return op(audio, sr)

def augment_and_mix(audio, width=3, depth=-1, alpha=1):
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    
    mix = np.zeros_like(audio)
    for i in range(width):
        audio_aug = audio.clone()
        d = depth if depth > 0 else np.random.randint(1,4)
        for _ in range(d):
            op = np.random.choice(augmentation.augmentations)
            audio_aug = apply_op(audio, op)
        mix += ws[i] * audio_aug 
    
    return (1 - m) * audio + m * mix 


class AugMixDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __getitem__(self, i):
        x, y = self.dataset[i]
        
        im_tuple = (x, augment_and_mix(x), augment_and_mix(x))
        return im_tuple, y 
    
    def __len__(self):
        return len(self.dataset)


def train(model, epochs, train_loader, optimizer, scheduler, device):
    model.train()
    
    for i in range(epochs):
        loss_ema = 0. 
        for _, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            images_all = torch.cat(images, 0).to(device)
            targets = targets.to(device)
            
            logits_all = model(images_all)
            logits_clean, logits_aug1, logits_aug2 = torch.split(
                logits_all, images[0].size(0)
            )
            
            loss = F.cross_entropy(logits_clean, targets)
            
            p_clean = F.softmax(logits_clean, dim=1)
            p_aug1 = F.softmax(logits_aug1, dim=1)
            p_aug2 = F.softmax(logits_aug2, dim=1)
            
            loss += jsd(p_clean, p_aug1, p_aug2, lamb=12)
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_ema += loss.item()
        print(f"{i+1} epoch: {loss_ema}")


def jsd(logits1, logits2, logits3, lamb=12):
    logits_mixture = (logits1, logits2, logits3) / 3.
    loss += lamb * (F.kl_div(logits_mixture, logits1, reduction='batchmean') +
                  F.kl_div(logits_mixture, logits2, reduction='batchmean') +
                  F.kl_div(logits_mixture, logits3, reduction='batchmean')
    ) / 3.
    return loss

if __name__ == "__main__":
    signal = torch.randn(16000)
    augmented_output = augment_and_mix(signal)


