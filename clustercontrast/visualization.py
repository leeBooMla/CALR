import os

import matplotlib.pyplot as plt
import torch
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import os.path as osp
from collections import OrderedDict
import matplotlib.image as mpimg
import matplotlib.patches as patches
from PIL import Image
from scipy.stats import mode


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def visualization(x,y,y2):
    #x, y = to_numpy(x), to_numpy(y)
    
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(x)
    #X_pca = PCA(n_components=2).fit_transform(x)

    ckpt_dir="images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, s=2, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=2, label="PCA")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y2, s=2, label="t-SNE")
    plt.legend()
    plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()

def visualization_cam(x,yc,yp,yf):
    #x, y = to_numpy(x), to_numpy(y)
    
    X_tsne = TSNE(n_components=2,random_state=33).fit_transform(x)
    cam = (mode(yc)).mode
    cam = int(cam)
    indices = np.where(yc == cam)
    local_pid, global_pid = [], []
    #for index in indices[0]:
    for index in range(len(yc)):
        cam = yc[index]
        global_pid.append(yp[index])
        fname = yf[index]
        local_dir  = osp.join('images/msmt17/cam'+str(cam), 'pseudo_label.npy')
        local_pseudo_dataset = np.load(local_dir)
        local_pseudo_dataset = local_pseudo_dataset.tolist()
        for f, pid, _ in sorted(local_pseudo_dataset):
            #f = f[:5]+'1'+f[5:]
            if f == fname:
                local_pid.append(pid)

    #X_pca = PCA(n_components=2).fit_transform(x)
    '''pseudo_labels_cam = OrderedDict()
    fname_cam = {}
    local_pid = []
    for i in range(len(x)):
        cam = yc[i]
        fname = yf[i]
        local_dir  = osp.join('images/market/cam'+str(cam+1), 'pseudo_label.npy')
        local_pseudo_dataset = np.load(local_dir)
        local_pseudo_dataset = local_pseudo_dataset.tolist()
        fname_cam[cam] = []
        pseudo_labels_cam[cam] = []
        for f, pid, cid in sorted(local_pseudo_dataset):
            f = f[:5]+'1'+f[5:]
            if f == fname:
                pseudo_labels_cam[cam].append(pid)
                local_pid.append(pid)
    local_pid = np.array(local_pid)'''

    ckpt_dir="images"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab10')  # 选择颜色映射，这里使用 'tab10'，可以根据需要更换

    # 创建相机标签到颜色的映射字典
    label_color_map = {label: cmap(i) for i, label in enumerate(set(yc))}
    label_color_map1 = {label: cmap(i) for i, label in enumerate(set(global_pid))}
    label_color_map2 = {label: cmap(i) for i, label in enumerate(set(local_pid))}

    fig, ax = plt.subplots()
    plt.figure(dpi= 512)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    '''ax.spines['top'].set_color('none')    # 隐藏顶部边框
    ax.spines['bottom'].set_color('none') # 隐藏底部边框
    ax.spines['left'].set_color('none')   # 隐藏左侧边框
    ax.spines['right'].set_color('none')  # 隐藏右侧边框'''
    fig1, ax1 = plt.subplots()
    plt.figure(dpi= 512)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 0.5)
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig2, ax2 = plt.subplots()
    plt.figure(dpi= 512)
    ax2.set_xlim(0, 0.5)
    ax2.set_ylim(0, 0.5)
    ax2.set_xticks([])
    ax2.set_yticks([])

    X_tsne[:,0] = (X_tsne[:,0] - min(X_tsne[:,0]))/(max(X_tsne[:,0]) - min(X_tsne[:,0]))
    X_tsne[:,1] = (X_tsne[:,1] - min(X_tsne[:,1])) * 0.5/(max(X_tsne[:,1]) - min(X_tsne[:,1]))

    count = 0
    for i, (x, y) in enumerate(X_tsne):
        image_path = yf[i]
        #image = mpimg.imread(image_path)
        image = Image.open('/data1/lpn/dataset/msmt17/MSMT17_V1/'+image_path)
        resized_image = image.resize((128, 256))
        resized_image_array = np.array(resized_image)
        camera_label = yc[i]
        color = label_color_map[camera_label]
        ax.imshow((resized_image_array), extent=(x-0.0125, x+0.0125, y-0.025, y+0.025), alpha=0.8)
        ax.add_patch(plt.Rectangle((x-0.0125, y-0.025), 0.025, 0.05, color=color, fill=False, lw=0.5))
        #ellipse = patches.Ellipse((x, y), 0.025, 0.05, angle=0, color= color1, fill=False, linewidth=0.5)
        #ax.add_patch(ellipse)
        count = i
        color1 = label_color_map1[global_pid[count]]
        ax1.imshow((resized_image_array), extent=(x-0.0125, x+0.0125, y-0.025, y+0.025), alpha=0.8)
        if global_pid[count] != -1:
            ax1.add_patch(plt.Rectangle((x-0.0125, y-0.025), 0.025, 0.05, color=color1, fill=False, lw=0.5))
        color2 = label_color_map2[local_pid[count]]
        ax2.imshow((resized_image_array), extent=(x-0.0125, x+0.0125, y-0.025, y+0.025), alpha=0.8)
        if local_pid[count] != -1:
            ax2.add_patch(plt.Rectangle((x-0.0125, y-0.025), 0.025, 0.05, color=color2, fill=False, lw=0.5))
        '''if count >= len(indices[0]):
            continue
        elif i == indices[0][count]:
            count += 1'''
    fig.savefig('images/cam.png', dpi = 512)
    fig1.savefig('images/global_pid.png', dpi = 512)
    fig2.savefig('images/local_pid.png', dpi = 512)
    plt.show()



    '''plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=yp, s=2, label="t-SNE")
    plt.legend()
    plt.subplot(122)
    #plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, s=2, label="PCA")
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=yc/10, s=2, label="t-SNE")
    plt.legend()
    plt.savefig('images/digits_tsne-pca.png', dpi=120)
    plt.show()'''

def evaluate_cluster(epochs,precision_global, recall_global, fscore_global,precision_global_b, recall_global_b, fscore_global_b, expansion_global,nmi_global, ave_precision_local,  ave_recall_local, ave_fscore_local, ave_precision_local_b,  ave_recall_local_b, ave_fscore_local_b, expansion_local, nmi_local, precision_global_refine, recall_global_refine, fscore_global_refine, precision_global_refine_b, recall_global_refine_b, fscore_global_refine_b, expansion_global_refine, nmi_refine):
    x = np.linspace(1, epochs, epochs)
    plt.rcParams['font.size'] = 12
    plt.figure(figsize=(8,4))
    plt.suptitle('pairwise')
    '''plt.subplot(2,2,1)
    plt.title('precision')
    plt.plot(x,precision_global,'r',label='precision_global')
    plt.plot(x,ave_precision_local,'g',label='precision_local')
    plt.plot(x,precision_global_refine,'b',label='precision_refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.subplot(2,2,2)
    plt.title('recall')
    plt.plot(x,recall_global,'r',label='recall_global')
    plt.plot(x,ave_recall_local,'g',label='precall_local')
    plt.plot(x,recall_global_refine,'b',label='recall_refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('recall')'''
    plt.subplot(1,2,1)
    #plt.title('F-Score')
    plt.plot(x,fscore_global,'r-',label='global')
    plt.plot(x,ave_fscore_local,'g-.',label='local')
    plt.plot(x,fscore_global_refine,'b:',label='refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('F-score')
    plt.subplot(1,2,2)
    #plt.title('NMI')
    plt.plot(x,nmi_global,'r-',label='global')
    plt.plot(x,nmi_local,'g-.',label='local')
    plt.plot(x,nmi_refine,'b:',label='refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('NMI')
    plt.savefig('images/msmt17/pairwise.png')

    plt.figure(figsize=(8,4))
    plt.suptitle('Bcubed')
    '''plt.subplot(2,2,1)
    plt.title('precision')
    plt.plot(x,precision_global_b,'r',label='precision_global')
    plt.plot(x,ave_precision_local_b,'g',label='precision_local')
    plt.plot(x,precision_global_refine_b,'b',label='precision_refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.subplot(2,2,2)
    plt.title('recall')
    plt.plot(x,recall_global_b,'r',label='recall_global')
    plt.plot(x,ave_recall_local_b,'g',label='precall_local')
    plt.plot(x,recall_global_refine_b,'b',label='recall_refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('recall')'''
    plt.subplot(1,2,1)
    #plt.title('F-Score')
    plt.plot(x,fscore_global_b,'r-',label='global')
    plt.plot(x,ave_fscore_local_b,'g-.',label='local')
    plt.plot(x,fscore_global_refine_b,'b:',label='refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('F-score')
    plt.subplot(1,2,2)
    #plt.title('Expansion')
    plt.plot(x,expansion_global,'r-',label='global')
    plt.plot(x,expansion_local,'g-.',label='local')
    plt.plot(x,expansion_global_refine,'b:',label='refine')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('expansion')
    plt.savefig('images/msmt17/bcubed.png')

def evaluate_local_cluster(epochs,precision_global, recall_global, fscore_global,precision_global_b, recall_global_b, fscore_global_b, expansion_global,nmi_global):
    x = np.linspace(1, epochs, epochs)
    plt.figure()
    plt.suptitle('pairwise')
    plt.subplot(2,2,1)
    plt.title('precision')
    plt.plot(x,precision_global,'r',label='precision')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.subplot(2,2,2)
    plt.title('recall')
    plt.plot(x,recall_global,'r',label='recall')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.subplot(2,2,3)
    plt.title('F-Score')
    plt.plot(x,fscore_global,'r',label='fscore')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('F-score')
    plt.subplot(2,2,4)
    plt.title('NMI')
    plt.plot(x,nmi_global,'r',label='NMI')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('NMI')
    plt.savefig('images/msmt17/cam15/pairwise.png')

    plt.figure()
    plt.suptitle('Bcubed')
    plt.subplot(2,2,1)
    plt.title('precision')
    plt.plot(x,precision_global_b,'r',label='precision')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('precision')
    plt.subplot(2,2,2)
    plt.title('recall')
    plt.plot(x,recall_global_b,'r',label='recall')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('recall')
    plt.subplot(2,2,3)
    plt.title('F-Score')
    plt.plot(x,fscore_global_b,'r',label='fscore')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('F-score')
    plt.subplot(2,2,4)
    plt.title('Expansion')
    plt.plot(x,expansion_global,'r',label='expansion')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('expansion')
    plt.savefig('images/msmt17/cam15/bcubed.png')