
# coding: utf-8

# # 9기 과제 -  딥러닝 기반 이미지 검색 엔진 개발

# ## 과제 개요

#  * 출제자 : 남상협(justin@buzzni.com) / 버즈니 (http://buzzni.com) 대표
#  * 배경 : 최근 들어서 딥러닝 기술의 발달로 이미지 분류 성능이 크게 향상 되었다. 이미지 분류 성능의 경우에는 컴퓨터가 이미 사람보다 더 뛰어난 성능을 보여주고 있다. 이런 기술 발달에 힘입어서 이미지 인식 검색 기술 성능도 함께 향상되었고 여러 기업에서 관련 서비스를 내놓고 있다. 이미지 분류 성능은 사람보다 컴퓨터가 더 뛰어난 성능을 보여주지만, 이미지 인식 검색 기술은 아직 사람과 컴퓨터 간의 격차가 매우 큰 도전적인 기술 분야이다.  본 과제는 아직 향상 시킬 여지가 많고 사회에서도 활용가능성이 무궁무진한 이미지 인식 검색에 대해서 기본 코드를 제공하고, 이를 기반으로 해서 더 높은 성능을 내는 고도화된 엔진을 만드는 것을 목표로 한다. 
#  * 활용 사례 : https://www.youtube.com/watch?v=-LsenqBcG8w
#  
# 

# ## 입력/출력
#  * 용어 설명 
#   * query 이미지 - 쇼핑몰 상품 상세에 판매자가 올린 상품 이미지 
#   * compare 이미지 - 사용자가 직접 찍어서 올린 리뷰 이미지 (좀더 많은 노이즈 포함)
#   
#  * 입력 : query 이미지
#  * 출력 : compare 이미지 중에서 query 이미지 상품과 가장 유사한 상품 TOP 10 (유사한 순서로 정렬)
#  
# ## 목표  
#  * 목표 : 다른 스타일의 이미지인 정제된 판매자 상품 이미지와, 노이즈가 많은 사용자 업로드 이미지 간에 같은 상품을 찾는 검색엔진을 만들게 된다.

# ## 평가 항목 
#  * test160 mode 성능평가 (70%)
#  * 제출문서 및 코드 (30%)
# ### 주의 사항
#  * 외부 업체에서 제공해주는 API (Google Cloud Vision API 나 기타 유사한 API) 를 사용하는 경우 0점 
#  * 다른 멘티의 코드를 카피한 경우 둘다 0점 
#  * 비정상적인 방법으로 문제를 해결한 경우 0점 (예를 들어서 평가 데이터를 모두 눈으로 확인해서, 사람이 직접 정답 데이터를 만드는등.)
#  * 제출한 문서 및 코드와, 제출한 성능평가가 다른 경우 0점 

# ## 제출 항목
#  * name - 자신의 이름을 넣는다. 실제 점수판(leader board)에는 공개가 안됨, 추후 평가시에 일치하는 이름의 멘티 점수로 사용함. 요청한 평가 중에서 가장 높은 점수의 평가 점수로 업데이트됨.
#  * nickname - 점수판에 공개되는 이름, 자신의 이름으로 해도 되고, 닉네임으로 해도 됨. (이상한 닉네임으로 할 경우 삭제)
#  * pred_result - 아래에 나온 base 코드를 참고해서, 자신이 만든 모델로 각 query 별로 가장 유사한 TOP 10 개의 compare 이미지 리스트를 만든 값
#  * mode - eval / test160, 파라미터 최적화를 할때에는 eval 사용 (횟수 제한 없음). 실제 평가에 사용하는 점수는 test160 모드 (한 ip 당 1일 1회 제한), 평가를 위해서 test160 mode 제출은 필수 
#   * eval mode 예 
# ```python
# name = '남상협'
# nickname = 'justin'
# email = 'justin@buzzni.com'
# mode = 'eval'
# r = requests.post('http://115.68.223.177:31000', json={"pred_result": system_result_dict,'name':name, 'nickname':nickname, 'mode':mode,'email':email})
# print (r.json())
# ```
#   * test160 mode 예 
# ```python
# name = '남상협'
# nickname = 'justin'
# email = 'justin@buzzni.com'
# mode = 'test160'
# r = requests.post('http://115.68.223.177:31000', json={"pred_result": system_result_dict,'name':name, 'nickname':nickname, 'mode':mode,'email':email})
# print (r.json())
# ```
# 
# ## 리턴값 
#  * ndcg 로 평가한 평가 점수, 1점 만점. 높을수록 좋은 점수
#  
# ## 점수 확인판
#  * http://eval.buzzni.net:31000/leader_board
#  
#  

# ## 성능 향상 포인트
#  1. 예제로 제공한 Resnet18 보다 더 깊은 딥러닝 모델을 사용한다. 
#  2. 예제로 제공한 딥러닝 모델보다 더 성능이 좋은 모델을 찾아서 사용한다.
#  3. 예제에서는 fc layer 에 있는 값을 사용했는데, 이 이전에 있는 layer 값들을 활용한다.
#  4. 이 이전에 있는 layer 값들을 더 잘 조합하는 방법을 스스로 생각하거나, 논문을 찾거나, 코드를 찾아서 적용해본다.
#  5. object detection 방법을 활용한다.

# In[1]:


import torch
import torch.nn as nn
import math


# In[2]:


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


# In[3]:


model_urls = {
    'resnet152': '/app/workspace/model/resnet152-b121ed2d.pth'
}


# In[4]:


# reference : https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_model = torch.load(model_urls['resnet152'])
        model.load_state_dict(pretrained_model)
    return model


# In[5]:


res152 = resnet152(pretrained=True)


# In[6]:


from IPython.core.display import display
from IPython.core.display import Image as Image2
from torchvision import models, transforms
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from PIL import Image
import requests

from io import BytesIO


# ### eval 모드 
#  * 모델을 사용하기전에는 eval() 모드를 호출해줘야한다.

# In[7]:


_ = res152.eval()


# ### 이미지 분류기 테스트 
#  * 각 이미지별로 어떤 카테고리로 분류하는지 테스트한다.

# In[8]:


trans = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # from http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
])

url = 'https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/'       'raw/596b27d23537e5a1b5751d2b0481ef172f58b539/imagenet1000_clsid_to_human.txt'

imagenet_classes = eval(requests.get(url).content)

images = [('cat', 'https://www.wired.com/wp-content/uploads/2015/02/catinbox_cally_by-helen-haden_4x31-660x495.jpg'),
          ('pomeranian', 'https://c.photoshelter.com/img-get/I0000q_DdkyvP6Xo/s/900/900/Pomeranian-Dog-with-Ball.jpg'),
          ('car', 'https://www.autocar.co.uk/sites/autocar.co.uk/files/styles/gallery_slide/public/images/car-reviews/first-drives/legacy/porsche-911_0.jpg')]
# images = [('shoes','http://thum.buzzni.com/unsafe/320x320/center/smart/http://172.16.10.6/maestro_img/query/query_21.jpg')]

for class_name, image_url in images:
    print(class_name)
    response = requests.get(image_url)
    im = Image.open(BytesIO(response.content))
    tens = Variable(trans(im))
    tens = tens.view(1, 3, 224, 224)
    preds = nn.LogSoftmax()(res152(tens)).data.cpu().numpy()
    res = np.argmax(preds)
    print('true (likely) label:', class_name)
    print('predicted', imagenet_classes[res], '\n')
    display(Image2(url=image_url,width=100))


# In[9]:


from glob import glob
import io


# ## Eval 데이터에 대한 결과 전송 
# 
# ### eval 데이터에 대해서 resnet18 모델을 사용해서 feature 를 추출한다.
# 

# In[10]:


EVAL_ROOT_DIR = '/app/workspace/data/eval/'


# ### query 이미지 확인

# In[11]:


ct = 0
for each in glob(EVAL_ROOT_DIR + "/query/*"):
    print (each)
    display(Image2(filename=each,width=100))
    ct+=1
    if ct > 3:
        break


# In[12]:


img_feature_dict = {}


# In[13]:


for folder in ['compare','query']:
    idx = 0
    for each in glob(EVAL_ROOT_DIR + "/%s/*"%(folder)):        
        fname = each.split("/")[-1]
        if fname in img_feature_dict:
            continue
        try:
            idx +=1
            print(idx)
            byteImgIO = io.BytesIO()
            byteImg = Image.open(each)
            byteImg.save(byteImgIO, "PNG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()
            dataBytesIO = io.BytesIO(byteImg)
            tens = Image.open(dataBytesIO)
            tens = Variable(trans(tens))
            tens = tens.view(1, 3, 224, 224)

            preds = nn.LogSoftmax()(res152(tens)).data.cpu().numpy()

            img_feature_dict[fname] = preds
        except Exception as e2:
            print(e2, fname)


# In[14]:


import operator
from scipy.spatial.distance import cosine


# ### 추출된 image feature 를 활용해서 각 query 별로 가장 거리가 가까운 이미지들을 찾는다.

# In[15]:


system_result_dict = {}
for each in glob(EVAL_ROOT_DIR + "/query/*"):
    
    fname = each.split("/")[-1]
    score_dict = {}
    print (each,fname)
    for other in glob(EVAL_ROOT_DIR + "/compare/*"):
        fname2 = other.split("/")[-1]
#         dist = cosine(res152_img_feature_dict[fname], res152_img_feature_dict[fname2])
        dist = cosine(img_feature_dict[fname], img_feature_dict[fname2])
#         print dist
        score_dict[fname2] = dist
#         break
    sorted_list = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=False)
    qid = fname.split("_")[-1].split(".")[0]
    system_result_dict[qid] = list(map(lambda i : i[0], sorted_list[:20]))


# ###  결과를 서버에 접수한다.
#  * name : 자신의 이름을 작성한다.
#  * nickname : leaderboard 에 올라갈 이름을 적는다. 
#  * email : 자신의 이메일 
#  * mode : eval 
#  * 결과 확인 : http://eval.buzzni.net:31000/leader_board?mode=eval

# In[16]:


name = '정다비치'
nickname = 'davichiar'
email = 'ardabitchy02@naver.com'
mode = 'eval'
r = requests.post('http://115.68.223.177:31000', json={"pred_result": system_result_dict,'name':name, 'nickname':nickname, 'mode':mode,'email':email})
print (r.json())


# ## Test160 데이터에 대한 결과 전송 (최종 평가)
#  * 하루에 한 ip 당 1번만 전송 가능함.
#  * 이전에 eval 을 통해서 충분히 최적화 한다음에 test 에 요청 

# In[17]:


TEST_ROOT_DIR = '/app/workspace/data/test160/'


# ### query 이미지 확인

# In[18]:


ct = 0
for each in glob(TEST_ROOT_DIR + "/query/*"):
    print (each)
    display(Image2(filename=each,width=100))
    ct+=1
    if ct > 3:
        break


# In[19]:


img_feature_dict = {}


# ### Test160 데이터에 대해서 resnet18 모델을 사용해서 feature 를 추출한다.

# In[20]:


for folder in ['compare','query']:
    idx = 0 
    for each in glob(TEST_ROOT_DIR + "/%s/*"%(folder)):        
        fname = each.split("/")[-1]
        idx +=1 
        if fname in img_feature_dict:
            continue
        try:
            
            print (idx)
            byteImgIO = io.BytesIO()
            byteImg = Image.open(each)
            byteImg.save(byteImgIO, "PNG")
            byteImgIO.seek(0)
            byteImg = byteImgIO.read()
            dataBytesIO = io.BytesIO(byteImg)
            tens = Image.open(dataBytesIO)
#             break
            tens = Variable(trans(tens))
            tens = tens.view(1, 3, 224, 224)

            preds = nn.LogSoftmax()(res152(tens)).data.cpu().numpy()

            img_feature_dict[fname] = preds
        except Exception as e2:
            print(e2, fname)


# ### 추출된 image feature 를 활용해서 각 query 별로 가장 거리가 가까운 이미지들을 찾는다.

# In[21]:


system_result_dict = {}
idx = 0
for each in glob(TEST_ROOT_DIR + "/query/*"):    
    fname = each.split("/")[-1]
    score_dict = {}
    idx+=1
    print (idx, fname)    
    for other in glob(TEST_ROOT_DIR + "/compare/*"):
        fname2 = other.split("/")[-1]
        dist = cosine(img_feature_dict[fname], img_feature_dict[fname2])
        score_dict[fname2] = dist
    sorted_list = sorted(score_dict.items(), key=operator.itemgetter(1), reverse=False)
    qid = fname.split("_")[-1].split(".")[0]
    system_result_dict[qid] = list(map(lambda i : i[0], sorted_list[:20]))


# ###  결과를 서버에 접수한다.
#  * name : 자신의 이름을 작성한다.
#  * nickname : leaderboard 에 올라갈 이름을 적는다. 
#  * email : 자신의 이메일 
#  * mode : test160 
#  * 결과 확인 : http://eval.buzzni.net:31000/leader_board?mode=test160

# In[22]:


name = '정다비치'
nickname = 'davichiar'
email = 'ardabitchy02@naver.com'
mode = 'test160'
r = requests.post('http://115.68.223.177:31000', json={"pred_result": system_result_dict,'name':name, 'nickname':nickname, 'mode':mode,'email':email})
print (r.json())

