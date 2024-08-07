import torch
import torch.nn.functional

class Predictor():
    def __init__(self, model, transform):
        self.model = model
        self.classes_en = ['adonis', 'african giant swallowtail', 'american snoot', 'an 88', 'appollo', 'atala', 'banded orange heliconian', 'banded peacock',
                           'beckers white', 'black hairstreak', 'blue morpho', 'blue spotted crow', 'brown siproeta', 'cabbage white', 'cairns birdwing', 'checquered skipper',
                           'chestnut', 'cleopatra', 'clodius parnassian', 'clouded sulphur', 'common banded awl', 'common wood-nymph', 'copper tail', 'crecent', 'crimson patch', 
                           'danaid eggfly', 'eastern coma', 'eastern dapple white', 'eastern pine elfin', 'elbowed pierrot', 'gold banded', 'great eggfly', 'great jay', 
                           'green celled cattleheart', 'grey hairstreak', 'indra swallow', 'iphiclus sister', 'julia', 'large marble', 'malachite', 'mangrove skipper', 
                           'mestra', 'metalmark', 'milberts tortoiseshell', 'monarch', 'mourning cloak', 'orange oakleaf', 'orange tip', 'orchard swallow', 'painted lady', 
                           'paper kite', 'peacock', 'pine white', 'pipevine swallow', 'popinjay', 'purple hairstreak', 'purplish copper', 'question mark', 'red admiral', 
                           'red cracker', 'red postman', 'red spotted purple', 'scarce swallow', 'silver spot skipper', 'sleepy orange', 'sootywing', 'southern dogface', 
                           'straited queen', 'tropical leafwing', 'two barred flasher', 'ulyses', 'viceroy', 'wood satyr', 'yellow swallow tail', 'zebra long wing']
    
        self.n_classes = len(self.classes_en)
        self.transform = transform
        
    #이미지를 예측하는 함수, 각 클래스 이름과 예측확룰을 튜플로 묶은 후 이들의 리스트를 만들어서 반환
    def predict(self, img):
        input_tensor = self.transform(img)
        input_batch = input_tensor.unsqueeze(0)  # 배치 차원 추가

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        input_batch = input_batch.to(device)

        self.model.eval()  # 모델을 평가 모드로 설정
        with torch.no_grad():
            output = self.model(input_batch)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        return self.classes_en[torch.argmax(probabilities)]
        
    


