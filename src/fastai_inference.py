from fastai.vision import *
from fastai.vision.data import normalize, imagenet_stats
from pathlib import Path

def instantiate_model(path):
    p = Path(path)
    return load_learner(p.parent, p.name)

def run_inference(model, frame, class_index):
    mean, std = torch.FloatTensor(imagenet_stats[0]), torch.FloatTensor(imagenet_stats[1])
    image_tensor = torch.from_numpy(frame).reshape([3] + list(frame.shape[0:2])).float()
    # normalizes the tensor around image net mean and std
    normalized = normalize(image_tensor, mean, std)

    img = Image(normalized)
    # returns all pytorch tensors: 
    # -> textual category, an index for the probability tensor, the probability tensor
    txt_cat, idx, prob  = model.predict(img)
    return prob[class_index].numpy()