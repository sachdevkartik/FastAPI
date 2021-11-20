# Code adopted from:
# https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/blob/b6e95ef873882ca9176537393703c67ca1a774a3/Chapter17/fmnist.py

from torch_snippets import *
from torchvision import datasets, models, transforms

PATH = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(PATH, "models" )
MODEL =  os.path.join(model_folder, "best_model.pt" )

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = torch.load(MODEL, map_location=torch.device(device))

class FMNIST(nn.Module):
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 
    'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    def __init__(self):
        super().__init__()
        self.model = models.resnet50(pretrained=True).to(device)    
        
        for param in self.model.parameters():
            param.requires_grad = False   
        self.model.fc = nn.Sequential(
                    nn.Linear(2048, 128),
                    nn.ReLU(inplace=True),
                    nn.Linear(128, 10)).to(device)

        self.model = nn.Sequential(
            nn.Linear(28 * 28, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        ).to(device)
        self.model.load_state_dict(torch.load('fmnist.weights.pth', map_location=device))
        logger.info('Loaded FMNIST Model')

    def forward(self, x):
        x = x.view(1, -1).to(device)
        pred = self.model(x)
        pred = F.softmax(pred, -1)[0]
        conf, clss = pred.max(-1)
        clss = self.classes[clss.cpu().item()]
        return conf.item(), clss

    def predict_from_path(self, path):
        x = cv2.imread(path,0)
        return self.predict_from_image(x)

def predict_from_image(image):
    image = np.array(image)
    image = cv2.resize(image, (28,28))
    
    # image = torch.Tensor(255 - image)/255.
    
    img = image/255.
    # img = img.view(28, 28)
    img3 = torch.Tensor(image).view(-1,1,28,28).to(device)
    np_output = model(img3).cpu().detach().numpy()
    pred = np.exp(np_output)/np.sum(np.exp(np_output))
    
    return {'prediction': pred}
    # 
    # 
    # x = torch.Tensor(255 - x)/255.
    # img = torch.Tensor(x).view(-1,1,28,28).to(device)
    # conf, clss = self(x)
    
