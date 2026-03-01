import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import pydicom
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from tqdm import tqdm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import timm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from collections import Counter
import torch.cuda.amp as amp  # 导入自动混合精度
from torch.optim.swa_utils import AveragedModel, update_bn  # 导入随机权重平均
import copy
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties

# 设置中文字体，避免中文乱码
font_path = '/System/Library/Fonts/STHeiti Light.ttc'
font_prop = FontProperties(fname=font_path)
rcParams['font.sans-serif'] = [font_prop.get_name()]  # 使用自定义字体
rcParams['axes.unicode_minus'] = False  # 正常显示负号


# 忽略警告
warnings.filterwarnings('ignore')

# 尝试设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("设置中文字体失败，继续使用默认字体")

# 设置随机种子，确保结果可复现
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 增强版DICOM处理函数
def read_dicom(path):
    """读取DICOM文件并转换为PIL图像，增强对比度"""
    try:
        dcm = pydicom.dcmread(path)
        img_array = dcm.pixel_array
        
        # 标准化像素值到0-255
        if img_array.max() > 0:
            img_array = img_array / img_array.max() * 255
        img_array = img_array.astype(np.uint8)
        
        # 如果是单通道，使用CLAHE增强对比度
        if len(img_array.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_array = clahe.apply(img_array)
            img = Image.fromarray(img_array).convert('RGB')
        else:
            img = Image.fromarray(img_array)
        
        return img
    except Exception as e:
        print(f"读取DICOM文件错误: {path}, 错误: {e}")
        return None

# 自定义数据集类
class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, mode='train'):
        """
        初始化乳腺癌数据集
        
        参数:
            root_dir (string): 数据集根目录
            transform (callable, optional): 可选的图像变换
            mode (string): 'train' 或 'test'
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        
        # 类别标签映射
        self.class_to_idx = {
            'BENIGN': 0,                 # 良性
            'MALIGNANT': 1,              # 恶性
            'BENIGN_WITHOUT_CALLBACK': 2 # 无需回访的良性
        }
        
        self.samples = self._make_dataset()
        
        # 获取各类别样本数量
        self.class_counts = {}
        for _, label in self.samples:
            self.class_counts[label] = self.class_counts.get(label, 0) + 1
        print(f"{mode}集各类别样本数量: {self.class_counts}")
        
        # 检查样本数量是否为空
        if len(self.samples) == 0:
            raise ValueError(f"未找到任何样本。请检查数据路径: {os.path.join(root_dir, mode + '集')}")
        
    def _make_dataset(self):
        """构建数据集路径和标签列表"""
        samples = []
        
        mode_dir = os.path.join(self.root_dir, self.mode + '集')
        if not os.path.exists(mode_dir):
            raise ValueError(f"数据集目录不存在: {mode_dir}")
            
        # 遍历所有类别目录
        for class_name in os.listdir(mode_dir):
            if class_name.startswith('.'):  # 跳过隐藏文件
                continue
                
            class_dir = os.path.join(mode_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            if class_name not in self.class_to_idx:
                print(f"警告: 跳过未知类别 {class_name}")
                continue
                
            class_idx = self.class_to_idx.get(class_name)
            
            # 遍历每个病例目录
            for patient_dir in os.listdir(class_dir):
                if patient_dir.startswith('.'):
                    continue
                    
                full_patient_dir = os.path.join(class_dir, patient_dir)
                if not os.path.isdir(full_patient_dir):
                    continue
                
                # 递归查找所有DICOM文件
                dicom_files_found = False
                for root, _, files in os.walk(full_patient_dir):
                    for file in files:
                        if file.endswith('.dcm'):
                            # 我们主要使用实际图像文件(通常是1-2.dcm)，而不是ROI掩码
                            if '-2.dcm' in file:
                                img_path = os.path.join(root, file)
                                samples.append((img_path, class_idx))
                                dicom_files_found = True
                
                if not dicom_files_found:
                    print(f"警告: 未在 {full_patient_dir} 中找到DICOM文件")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if idx >= len(self.samples):
            raise IndexError(f"索引 {idx} 超出范围 (样本数量: {len(self.samples)})")
            
        img_path, class_idx = self.samples[idx]
        
        # 读取并处理DICOM图像
        try:
            image = read_dicom(img_path)
            
            if image is None:
                # 如果图像读取失败，返回一个黑色图像
                print(f"警告: 图像读取失败，使用黑色图像替代: {img_path}")
                image = Image.new('RGB', (320, 320), color=0)
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return image, class_idx
        except Exception as e:
            print(f"获取样本出错: {e}, 索引: {idx}, 路径: {img_path}")
            # 返回一个黑色图像和类别标签
            image = Image.new('RGB', (320, 320), color=0)
            if self.transform:
                image = self.transform(image)
            return image, class_idx

# 定义增强版图像变换
def get_transforms(mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((384, 384)),  # 更大的输入尺寸
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),  # 增加垂直翻转概率
            transforms.RandomRotation(20),  # 增加旋转角度
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 更大的仿射变换
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),  # 增强颜色变化
            transforms.RandomAutocontrast(p=0.2),  # 添加自动对比度调整
            transforms.RandomEqualize(p=0.1),  # 添加自动均衡化
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # 添加高斯模糊
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),  # 添加透视变换
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1)  # 随机擦除，模拟遮挡
        ])
    else:
        return transforms.Compose([
            transforms.Resize((384, 384)),  # 更大的输入尺寸，与训练一致
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# 构建改进版模型 - 使用更强大的网络架构
class EnhancedBreastCancerModel(nn.Module):
    def __init__(self, model_name='efficientnet_b4', num_classes=3, pretrained=True):
        super(EnhancedBreastCancerModel, self).__init__()
        # 使用timm库加载预训练模型
        try:
            # 尝试下载预训练模型
            self.model = timm.create_model(model_name, pretrained=pretrained)
            print(f"成功加载预训练模型: {model_name}")
        except Exception as e:
            # 如果下载失败，尝试不使用预训练权重
            print(f"预训练模型加载失败: {e}")
            print("尝试不使用预训练权重创建模型...")
            try:
                self.model = timm.create_model(model_name, pretrained=False)
                print(f"成功创建模型: {model_name} (无预训练权重)")
            except Exception as e2:
                # 如果仍然失败，尝试使用更通用的模型
                print(f"模型创建失败: {e2}")
                print("尝试使用DenseNet201模型...")
                try:
                    self.model = timm.create_model('densenet201', pretrained=True)
                    model_name = 'densenet201'
                except Exception as e3:
                    print(f"DenseNet201模型创建失败: {e3}")
                    print("最终回退到ResNet50模型...")
                    self.model = models.resnet50(pretrained=pretrained)
                    model_name = 'resnet50'
        
        # 获取最后全连接层的输入特征数并替换分类器
        if 'efficientnet' in model_name:
            if hasattr(self.model, 'classifier'):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
            else:
                # 处理不同版本的efficientnet结构差异
                print(f"警告: 无法找到classifier属性，尝试其他方法替换分类器")
                if hasattr(self.model, 'head'):
                    in_features = self.model.head.in_features
                    self.model.head = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(in_features, 1024),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(1024),
                        nn.Dropout(0.5),
                        nn.Linear(1024, 512),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(512),
                        nn.Dropout(0.3),
                        nn.Linear(512, num_classes)
                    )
        elif 'densenet' in model_name:
            if hasattr(self.model, 'classifier'):
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
        else:  # 其他模型，如resnet
            if hasattr(self.model, 'fc'):
                in_features = self.model.fc.in_features
                self.model.fc = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(in_features, 1024),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.5),
                    nn.Linear(1024, 512),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(512),
                    nn.Dropout(0.3),
                    nn.Linear(512, num_classes)
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_name}，无法找到分类器层")
    
    def forward(self, x):
        return self.model(x)

# 使用焦点损失解决类别不平衡问题
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# 训练函数
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, scheduler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="训练")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        if scaler is not None:  # 使用混合精度训练
            # 前向传播
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:  # 常规训练
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
        
        # 如果使用OneCycleLR，需要在每个batch后更新
        if scheduler is not None:
            scheduler.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': running_loss / (progress_bar.n + 1),
            'acc': 100. * correct / total
        })
    
    return running_loss / len(dataloader), 100. * correct / total

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="验证")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': running_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
    
    # 计算指标
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return running_loss / len(dataloader), 100. * correct / total, precision, recall, f1, all_preds, all_labels

# 测试函数
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="测试"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    print(f"测试集准确率: {accuracy*100:.2f}%")
    print(f"测试集精准率: {precision*100:.2f}%")
    print(f"测试集召回率: {recall*100:.2f}%")
    print(f"测试集F1分数: {f1*100:.2f}%")
    print("混淆矩阵:")
    print(conf_matrix)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    
    classes = ['良性', '恶性', '无需回访良性']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 在格子中添加数字
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig('confusion_matrix.png')
    
    return accuracy, precision, recall, f1, conf_matrix

# 创建加权采样器，处理类别不平衡问题
def create_weighted_sampler(dataset):
    """
    为数据集创建加权采样器，使各类别样本在训练中均衡出现
    适用于原始数据集或Subset数据集
    """
    # 判断是否为Subset数据集
    if hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # 是Subset，需要通过indices获取对应的标签
        original_dataset = dataset.dataset
        indices = dataset.indices
        if hasattr(original_dataset, 'samples'):
            # 通过索引获取原始数据集中的标签
            targets = [original_dataset.samples[i][1] for i in indices]
        else:
            # 如果没有samples属性，尝试其他方式获取标签
            targets = []
            for i in indices:
                _, label = original_dataset[i]
                targets.append(label)
    else:
        # 原始数据集，直接获取标签
        if hasattr(dataset, 'samples'):
            targets = [label for _, label in dataset.samples]
        else:
            # 如果没有samples属性，则遍历数据集获取标签
            targets = []
            for _, label in dataset:
                targets.append(label)
    
    # 计算类别权重
    class_counts = Counter(targets)
    total_samples = len(targets)
    
    # 计算每个类别的权重（类别样本越少，权重越高）
    weights = {cls: total_samples / (count * len(class_counts)) for cls, count in class_counts.items()}
    
    # 为每个样本分配权重
    sample_weights = [weights[label] for label in targets]
    
    # 创建加权随机采样器
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

# 主函数
def main():
    # 设置随机种子
    set_seed(42)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 数据集路径
    data_root = "完美data"
    if not os.path.exists(data_root):
        raise ValueError(f"数据根目录不存在: {data_root}")
    
    # 创建数据集和数据加载器
    train_transform = get_transforms(mode='train')
    test_transform = get_transforms(mode='test')
    
    # 加载数据集
    try:
        print("正在加载训练集...")
        train_dataset = BreastCancerDataset(data_root, transform=train_transform, mode='训练')
        print("正在加载测试集...")
        test_dataset = BreastCancerDataset(data_root, transform=test_transform, mode='测试')
    except Exception as e:
        print(f"加载数据集失败: {e}")
        raise
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 检查训练集和测试集是否为空
    if len(train_dataset) == 0:
        raise ValueError("训练集为空")
    if len(test_dataset) == 0:
        raise ValueError("测试集为空")
    
    # 确保每个类别都有样本
    train_classes = set(train_dataset.class_counts.keys())
    test_classes = set(test_dataset.class_counts.keys())
    if len(train_classes) < 3:
        print(f"警告: 训练集中缺少类别，只有{train_classes}")
    if len(test_classes) < 3:
        print(f"警告: 测试集中缺少类别，只有{test_classes}")
    
    # 首先创建加权采样器处理整个训练集不平衡问题
    print("创建加权采样器...")
    full_train_sampler = create_weighted_sampler(train_dataset)
    
    # 训练集和验证集划分 (80% 训练, 20% 验证)
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    print(f"将训练集划分为: {train_size}训练样本, {val_size}验证样本")
    
    # 使用固定的随机种子进行分割以保证可复现性
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size], 
        generator=generator
    )
    
    # 为分割后的训练集创建加权采样器
    train_sampler = create_weighted_sampler(train_dataset)
    
    # 批量大小，根据设备调整
    batch_size = 8 if device.type == 'cpu' else 32  # 增大批量大小以提高效率
    print(f"使用批量大小: {batch_size}")
    
    # 使用worker数量: CPU核心数的一半或最小1
    import multiprocessing
    num_workers = max(1, min(4, multiprocessing.cpu_count() // 2))
    print(f"使用worker数量: {num_workers}")
    
    # 创建数据加载器，减少worker数量以提高稳定性
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # 创建多个不同的模型进行集成学习
    models_config = [
        {'name': 'efficientnet_b4', 'pretrained': True},
        {'name': 'densenet201', 'pretrained': True},
        {'name': 'resnet101', 'pretrained': True}
    ]
    
    ensemble_models = []
    
    for model_config in models_config:
        try:
            print(f"尝试创建{model_config['name']}模型...")
            model = EnhancedBreastCancerModel(
                model_name=model_config['name'], 
                num_classes=3, 
                pretrained=model_config['pretrained']
            ).to(device)
            ensemble_models.append(model)
        except Exception as e:
            print(f"{model_config['name']}模型创建失败: {e}")
    
    if not ensemble_models:
        print("所有模型创建失败，尝试创建基础模型...")
        model = models.resnet50(pretrained=True)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 3)
        model = model.to(device)
        ensemble_models = [model]
    
    print(f"成功创建{len(ensemble_models)}个模型进行集成")
    
    # 为每个模型定义训练参数和优化器
    model_trainers = []
    
    for i, model in enumerate(ensemble_models):
        # 定义不同的损失函数，增加多样性
        if i == 0:
            criterion = FocalLoss(alpha=1, gamma=2)
        elif i == 1:
            class_weights = torch.tensor([1.0, 2.0, 1.5], device=device)  # 根据类别重要性设置
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        
        # 定义不同的优化器，增加多样性
        if i == 0:
            optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
            scheduler = OneCycleLR(optimizer, max_lr=0.001, epochs=40, 
                                  steps_per_epoch=len(train_loader))
        elif i == 1:
            optimizer = optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        else:
            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=5, verbose=True
            )
        
        # 创建随机权重平均模型
        swa_model = AveragedModel(model)
        
        model_trainers.append({
            'model': model,
            'criterion': criterion,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'swa_model': swa_model,
            'best_val_loss': float('inf'),
            'history': {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'val_precision': [],
                'val_recall': [],
                'val_f1': []
            }
        })
    
    # 训练每个模型
    num_epochs = 40  # 增加轮数
    
    for epoch in range(num_epochs):
        print(f"\n===== 第 {epoch+1}/{num_epochs} 轮训练 =====")
        
        for i, trainer in enumerate(model_trainers):
            model = trainer['model']
            criterion = trainer['criterion']
            optimizer = trainer['optimizer']
            scheduler = trainer['scheduler']
            history = trainer['history']
            
            print(f"\n训练模型 {i+1}/{len(model_trainers)} ({type(model).__name__})")
            
            # 使用混合精度训练加速
            if device.type == 'cuda':
                scaler = amp.GradScaler()
                # 训练一个轮次（使用混合精度）
                with torch.cuda.amp.autocast():
                    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, 
                                                       scaler=scaler, scheduler=scheduler if i == 0 else None)
            else:
                # CPU训练
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # 更新非OneCycle学习率
            if i != 0 and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"当前学习率: {current_lr:.6f}")
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = validate(
                model, val_loader, criterion, device
            )
            
            # 更新ReduceLROnPlateau学习率调度器
            if i == 2:
                scheduler.step(val_loss)
            elif i == 1:
                scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1'].append(val_f1)
            
            # 打印结果
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%")
            print(f"验证精准率: {val_precision:.4f}, 验证召回率: {val_recall:.4f}, 验证F1分数: {val_f1:.4f}")
            
            # 保存最佳模型
            if val_loss < trainer['best_val_loss']:
                trainer['best_val_loss'] = val_loss
                torch.save(model.state_dict(), f'best_model_{i+1}.pth')
                print(f"保存模型{i+1}的最佳状态")
            
            # 更新SWA模型（在训练后期）
            if epoch >= num_epochs // 2:
                trainer['swa_model'].update_parameters(model)
                print(f"更新模型{i+1}的SWA权重")
        
        # 每10轮保存一次检查点
        if (epoch + 1) % 10 == 0:
            for i, trainer in enumerate(model_trainers):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': trainer['model'].state_dict(),
                    'optimizer_state_dict': trainer['optimizer'].state_dict(),
                    'scheduler_state_dict': trainer['scheduler'].state_dict() if hasattr(trainer['scheduler'], 'state_dict') else None,
                    'best_val_loss': trainer['best_val_loss'],
                    'history': trainer['history']
                }, f'checkpoint_model_{i+1}_epoch_{epoch+1}.pth')
            print(f"已保存所有模型的检查点: epoch_{epoch+1}")
    
    # 训练结束后，更新SWA模型的批归一化统计量
    print("\n更新SWA模型批归一化统计量...")
    for i, trainer in enumerate(model_trainers):
        update_bn(train_loader, trainer['swa_model'], device=device)
        # 保存SWA模型
        torch.save(trainer['swa_model'].state_dict(), f'swa_model_{i+1}.pth')
        print(f"保存SWA模型 {i+1}")
    
    # 加载每个模型的最佳状态用于集成
    print("\n准备测试集成模型...")
    best_models = []
    
    for i, trainer in enumerate(model_trainers):
        # 尝试加载最佳模型
        try:
            model = copy.deepcopy(trainer['model'])
            model.load_state_dict(torch.load(f'best_model_{i+1}.pth'))
            best_models.append(model)
            print(f"已加载模型{i+1}的最佳状态")
        except Exception as e:
            print(f"加载模型{i+1}失败: {e}")
            # 尝试加载SWA模型
            try:
                swa_model = trainer['swa_model']
                best_models.append(swa_model)
                print(f"已加载模型{i+1}的SWA状态")
            except Exception as e2:
                print(f"加载模型{i+1}的SWA状态失败: {e2}")
                # 使用当前模型
                best_models.append(trainer['model'])
                print(f"使用模型{i+1}的当前状态")
    
    # 测试每个单独模型
    all_model_metrics = []
    
    for i, model in enumerate(best_models):
        try:
            print(f"\n测试模型 {i+1}/{len(best_models)}:")
            accuracy, precision, recall, f1, _ = test_model(model, test_loader, device)
            all_model_metrics.append({
                'model_index': i+1,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
        except Exception as e:
            print(f"测试模型{i+1}失败: {e}")
    
    # 进行模型集成测试
    if len(best_models) > 1:
        try:
            print("\n测试集成模型:")
            ensemble_accuracy, ensemble_precision, ensemble_recall, ensemble_f1, ensemble_conf_matrix = test_ensemble(
                best_models, test_loader, device
            )
            
            # 显示集成结果
            print(f"\n集成模型测试结果:")
            print(f"准确率: {ensemble_accuracy*100:.2f}%")
            print(f"精准率: {ensemble_precision*100:.2f}%")
            print(f"召回率: {ensemble_recall*100:.2f}%")
            print(f"F1分数: {ensemble_f1*100:.2f}%")
            
            # 可视化混淆矩阵
            plt.figure(figsize=(10, 8))
            plt.imshow(ensemble_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('集成模型混淆矩阵')
            plt.colorbar()
            
            classes = ['良性', '恶性', '无需回访良性']
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # 在格子中添加数字
            thresh = ensemble_conf_matrix.max() / 2.
            for i in range(ensemble_conf_matrix.shape[0]):
                for j in range(ensemble_conf_matrix.shape[1]):
                    plt.text(j, i, format(ensemble_conf_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if ensemble_conf_matrix[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('真实标签')
            plt.xlabel('预测标签')
            plt.savefig('ensemble_confusion_matrix.png')
            
            # 比较所有模型性能
            model_names = [f"模型{m['model_index']}" for m in all_model_metrics]
            model_names.append("集成模型")
            
            accuracies = [m['accuracy']*100 for m in all_model_metrics]
            accuracies.append(ensemble_accuracy*100)
            
            precisions = [m['precision']*100 for m in all_model_metrics]
            precisions.append(ensemble_precision*100)
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(model_names))
            width = 0.35
            
            plt.bar(x - width/2, accuracies, width, label='准确率')
            plt.bar(x + width/2, precisions, width, label='精准率')
            
            plt.xlabel('模型')
            plt.ylabel('百分比 (%)')
            plt.title('各模型性能比较')
            plt.xticks(x, model_names)
            plt.ylim(0, 100)
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 在柱状图上显示具体数值
            for i, v in enumerate(accuracies):
                plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center')
            
            for i, v in enumerate(precisions):
                plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center')
            
            plt.tight_layout()
            plt.savefig('model_comparison.png')
            
            return ensemble_precision
        except Exception as e:
            print(f"集成测试失败: {e}")
            
            # 返回最佳单模型的精准率
            if all_model_metrics:
                best_metric = max(all_model_metrics, key=lambda x: x['precision'])
                return best_metric['precision']
            return 0.0
    else:
        # 只有一个模型时，返回该模型的精准率
        if all_model_metrics:
            return all_model_metrics[0]['precision']
        return 0.0


# 集成模型测试函数
def test_ensemble(models, test_loader, device):
    for model in models:
        model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="集成测试"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 存储所有模型的预测结果
            model_outputs = []
            
            # 获取每个模型的输出
            for model in models:
                outputs = model(inputs)
                model_outputs.append(outputs)
            
            # 将所有模型的输出取平均
            ensemble_outputs = torch.zeros_like(model_outputs[0])
            for output in model_outputs:
                ensemble_outputs += output
            ensemble_outputs /= len(models)
            
            # 获取最终预测
            _, predictions = ensemble_outputs.max(1)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='weighted')
    recall = recall_score(all_targets, all_predictions, average='weighted')
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    print(f"集成模型准确率: {accuracy*100:.2f}%")
    print(f"集成模型精准率: {precision*100:.2f}%")
    print(f"集成模型召回率: {recall*100:.2f}%")
    print(f"集成模型F1分数: {f1*100:.2f}%")
    print("集成模型混淆矩阵:")
    print(conf_matrix)
    
    return accuracy, precision, recall, f1, conf_matrix

if __name__ == "__main__":
    main() 