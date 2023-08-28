import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
# import skimage
import pdb

# print(clip.available_models())
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, download_root="./pretrained_models")
 
 
# descriptions = [f"{imagename}" for imagename in os.listdir("./test-images") if imagename.endwith(".png")] 

original_images = []
for imagename in os.listdir("./test-images"):
    # descriptions.append(imagename)
    image = Image.open(os.path.join('./test-images',imagename))
    original_images.append(torch.unsqueeze(preprocess(image),dim=0))

descriptions = ["car","tre","book","cloud","sky","building","Ace","flag","Sponge","Dice","Airplane","mother","SpongeBob","cartoon"]
# pdb.set_trace()
texts = clip.tokenize(descriptions).to(device)
images = torch.tensor(torch.cat(original_images))
# model.eval()



with torch.no_grad():
    image_features = model.encode_image(images)
    image_features /= image_features.norm(dim = -1,keepdim = True)
    text_features = model.encode_text(texts)
    text_features /= text_features.norm(dim=-1, keepdim=True)

text_probs = (100*image_features @ text_features.T).softmax(dim=-1)
top_probs, top_labels = text_probs.cpu().topk(3, dim=-1)


count = len(descriptions)

# pdb.set_trace()
plt.figure(figsize = (16, 16))

for i, image in enumerate(original_images):
    plt.subplot(4, 4, 2*i+1)
    plt.imshow(image.squeeze(0).permute(1,2,0))
    plt.axis("off")

    plt.subplot(4, 4, 2 * i + 2)
    y = np.arange(top_probs.shape[-1])
    plt.grid()
    plt.barh(y, top_probs[i])
    # get current axis
    # plt.gca().invert_yaxis()
    # plt.gca().set_axisbelow(True)
    plt.yticks(y, [descriptions[index] for index in top_labels[i].numpy()])
    plt.xlabel("probability")

plt.subplots_adjust(wspace = 0.5)
plt.savefig("./test.png")   
