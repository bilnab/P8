
#construction et rechargement du modèle

import tensorflow as tf
import json
import cv2
import numpy as np
import itertools
import cv2
import matplotlib.pyplot as plt
import random
import os
from tensorflow.python.keras.preprocessing import image
import time


os.environ["CUDA_VISIBLE_DEVICES"]="-1"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import keras
from keras.models import *
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, BatchNormalization, concatenate, Reshape, Activation

IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_LAST

if IMAGE_ORDERING == 'channels_last':
    pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
    MERGE_AXIS = -1

def get_vgg_encoder(input_height=224,  input_width=224, pretrained=False, channels=3): #pretrained='imagenet', channels=3):

    assert input_height % 32 == 0
    assert input_width % 32 == 0

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(channels, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=IMAGE_ORDERING)(x)
    f1 = x
    
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',  name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=IMAGE_ORDERING)(x)
    f5 = x

    if pretrained == 'imagenet':
        #telechargement du fichier de poids 
        #astuce pour extraire le nom en dernier de l'url avec pretrained_url.split("/")[-1]
        VGG_Weights_path = keras.utils.get_file(pretrained_url.split("/")[-1], pretrained_url)
        #chargement : https://keras.io/api/models/model_saving_apis/#loadweights-method
            #by_name=True: to load weights by name => necessite d avoir les memes noms de couches
        Model(img_input, x).load_weights(VGG_Weights_path, by_name=True, skip_mismatch=True)

    return img_input, [f1, f2, f3, f4, f5]

def vgg_unet(n_classes, input_height=416, input_width=608, encoder_level=3, channels=3):

    model = _unet(n_classes, get_vgg_encoder,input_height=input_height, input_width=input_width, channels=channels)
    model.model_name = "vgg_unet"
    return model

def _unet(n_classes, encoder, l1_skip_conn=True, input_height=416, input_width=608, channels=3):

    img_input, levels = encoder(input_height=input_height, input_width=input_width, channels=channels)
    [f1, f2, f3, f4, f5] = levels

    o = f4

    #rajoute du padding de 0 autour 
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(512, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    #normalisation de la sortie
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f3], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(256, (3, 3), padding='valid', activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)
    o = (concatenate([o, f2], axis=MERGE_AXIS))
    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(128, (3, 3), padding='valid' , activation='relu' , data_format=IMAGE_ORDERING))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(o)

    if l1_skip_conn:
        o = (concatenate([o, f1], axis=MERGE_AXIS))

    o = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(o)
    o = (Conv2D(64, (3, 3), padding='valid', activation='relu', data_format=IMAGE_ORDERING, name="seg_feats"))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same',data_format=IMAGE_ORDERING)(o)
    
    oh = Model(img_input, o).output_shape[1]
    ow = Model(img_input, o).output_shape[2]
    nc = Model(img_input, o).output_shape[3]
    ih = Model(img_input, o).input_shape[1]
    iw = Model(img_input, o).input_shape[2]
    
    #segmentation
    o = (Reshape((Model(img_input, o).output_shape[1]*Model(img_input, o).output_shape[2], -1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    
    model.output_height = oh
    model.output_width = ow
    model.n_classes = nc
    model.input_height = ih
    model.input_width = iw
    

    
    return model

def seg_prediction(model,adress,gt_adress):
    #lecture en couleur
    read_image_type=1
    #lecture en BGR
    im2p = cv2.imread(adress, read_image_type)

    #soft denoising
    #im2p = cv2.fastNlMeansDenoising(im2p, h=3)
    #equalization
    #im2p[:,:,0] = cv2.equalizeHist(im2p[:,:,0])
    #im2p[:,:,1] = cv2.equalizeHist(im2p[:,:,1])
    #im2p[:,:,2] = cv2.equalizeHist(im2p[:,:,2])
    #pour affichier plus tard en RGB
    im2show=im2p[:, :, ::-1]
    #resize width et height dans cet ordre pour etre coherent avec ce qu'on a fait avant
    im2p = cv2.resize(im2p, (model.input_width, model.input_height)).astype(np.float32)


    means = [103.939, 116.779, 123.68]

    for i in range(3):
        im2p[:, :, i] -= means[i]
    #passage en RGB
    im2p = im2p[:, :, ::-1]
    #on rajoute la dimension batch en first: on est oblige de mettre np.array pour faire le 4D sinon ca nous fait une liste de 3D
    im2p = np.array([im2p])
    #prediction en 3D (batch=1,height*weight,classes)
    #mieux vaut utiliser predict qui bosse en np plus que le model(x) directement qui bosse avec des tenseurs car problemes sur reshape par exemple
    im_p = model.predict(im2p)#model(im2p, training=False)
    im_p = model.predict(im2p)
    im_p = model.predict(im2p)
    im_p = model.predict(im2p)
    #reshaping avec prediction de la classe la plus probable avec argmax selon les classes => on passe en 2D
    im_p = im_p.reshape((model.output_height,  model.output_width, model.n_classes)).argmax(axis=2)
    
    #VISUALISATION DE LA PREDICTION
    #squelette RGB
    seg_img = np.zeros((model.output_height, model.output_width, 3))
    #colorisation des classes en RGB : les couleurs tirés aleatoirement seront les memes a cause du seed à 0
    random.seed(24)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(model.n_classes)]
    random.seed(24)
    colors2 = [(random.randint(0, 255)/255, random.randint(0, 255)/255, random.randint(0, 255)/255) for _ in range(model.n_classes)]
    for c in range(model.n_classes):
            seg_c = im_p[:, :] == c
            seg_img[:, :, 0] += ((seg_c)*(colors[c][0]))
            seg_img[:, :, 1] += ((seg_c)*(colors[c][1]))
            seg_img[:, :, 2] += ((seg_c)*(colors[c][2]))
    #resizing au dimension de l'image brute de depart
    original_h = im2show.shape[0]
    original_w = im2show.shape[1]
    seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    
    #COMPARAISON AVEC LE GROUND TRUTH
    #lecture en BGR 2D
    gt = cv2.imread(gt_adress, 0)
    #squelette 
    gt2c = np.zeros((gt.shape[0], gt.shape[1]))

    #dictionnaire de main categories
    cats = {'void': [0, 1, 2, 3, 4, 5, 6],
        'flat': [7, 8, 9, 10],
        'construction': [11, 12, 13, 14, 15, 16],
        'object': [17, 18, 19, 20],
        'nature': [21, 22],
        'sky': [23],
        'human': [24, 25],
        'vehicle': [26, 27, 28, 29, 30, 31, 32, 33, -1]}

    for i in range(-1, 34):
        if i in cats['void']:
            gt2c[gt==i]=0
        elif i in cats['flat']:
            gt2c[gt==i]=1
        elif i in cats['construction']:
            gt2c[gt==i]=2
        elif i in cats['object']:
            gt2c[gt==i]=3
        elif i in cats['nature']:
            gt2c[gt==i]=4
        elif i in cats['sky']:
            gt2c[gt==i]=5
        elif i in cats['human']:
            gt2c[gt==i]=6
        elif i in cats['vehicle']:
            gt2c[gt==i]=7

    #squelette RGB
    gt2c_color = np.zeros((gt.shape[0], gt.shape[1], 3))
    for c in range(model.n_classes):
        seg_c = gt2c[:, :] == c
        gt2c_color[:, :, 0] += ((seg_c)*(colors[c][0]))
        gt2c_color[:, :, 1] += ((seg_c)*(colors[c][1]))
        gt2c_color[:, :, 2] += ((seg_c)*(colors[c][2]))
        
    #LEGENDE
    #squelette de legende blanc(+255)
    legend = np.zeros((gt.shape[0], gt.shape[1]//4, 3),dtype="uint8") + 255
    #enumerate = compteur et zip 2 listes comme une fermeture eclair
    for (i, (class_name, color)) in enumerate(zip(list(cats.keys()),colors)):
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (20, (i * gt.shape[0]//8) + (gt.shape[0]//8)*2//3), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 4)
        cv2.rectangle(legend, ((gt.shape[1]//4)-50, (i * gt.shape[0]//8)), ((gt.shape[1]//4), (i * gt.shape[0]//8) + gt.shape[0]//8),tuple(color), -1)
        
    #AFFICHAGE
    f, (a0, a1, a2) = plt.subplots(1, 3,figsize=(20,5), gridspec_kw={'width_ratios': [4, 4, 4]})

    a0.title.set_text('image brute')
    a0.imshow(im2show)

    #a1.title.set_text("Légende")
    #a1.imshow(legend.astype('uint8'))

    a1.title.set_text("segmentation prediction")
    a1.imshow(seg_img.astype('uint8'))

    a2.title.set_text("ground thruth")
    a2.imshow(gt2c_color.astype('uint8'))
    
    f.savefig('static/result_image.png',bbox_inches='tight')
    #plt.show()
    
    #EVALUATION DE LA PREDICTION:
    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    tn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)
    
        #on size sur la sortie de modèle 
    gt2c_rs = cv2.resize(gt2c, (model.output_width, model.output_height), interpolation=cv2.INTER_NEAREST)
    prf = im_p.flatten()
    gtf = gt2c_rs.flatten()
    
    #boucles sur les 8 classes
    for cl_i in range(model.n_classes):
        #calcul des TP par classe
        tp[cl_i] += np.sum((prf == cl_i) * (gtf == cl_i))
        #calcul des FP par classe
        fp[cl_i] += np.sum((prf == cl_i) * (gtf != cl_i))
        #calcul des FN par classe
        fn[cl_i] += np.sum((prf != cl_i) * (gtf == cl_i))
        #calcul des TN par classe
        tn[cl_i] += np.sum((prf != cl_i) * (gtf != cl_i))
        #nombre de pixels par classe
        n_pixels[cl_i] += np.sum(gtf == cl_i)

    cl_IOU1 = (tp+0.000000000001) / (tp + fp + fn + 0.000000000001)
    cl_pct_pixel1 = n_pixels / np.sum(n_pixels)
    flat_IOU1 = np.sum(cl_IOU1*cl_pct_pixel1)
    mean_IOU1 = np.mean(cl_IOU1)
    
    #print("flat IOU sur le format ",model.output_width,'X',model.output_height," :","{:.2f}".format(flat_IOU1))
    #print("mean IOU sur le format ",model.output_width,'X',model.output_height," :","{:.2f}".format(mean_IOU1))
    
    #############
    
    tp = np.zeros(model.n_classes)
    fp = np.zeros(model.n_classes)
    fn = np.zeros(model.n_classes)
    tn = np.zeros(model.n_classes)
    n_pixels = np.zeros(model.n_classes)
    
        #on size sur la sortie de modèle 
    im_p_rs = cv2.resize(im_p, (gt2c.shape[1], gt2c.shape[0]), interpolation=cv2.INTER_NEAREST)
    prf = im_p_rs.flatten()
    gtf = gt2c.flatten()

    
    
    #boucles sur les 8 classes
    for cl_i in range(model.n_classes):
        #calcul des TP par classe
        tp[cl_i] += np.sum((prf == cl_i) * (gtf == cl_i))
        #calcul des FP par classe
        fp[cl_i] += np.sum((prf == cl_i) * (gtf != cl_i))
        #calcul des FN par classe
        fn[cl_i] += np.sum((prf != cl_i) * (gtf == cl_i))
        #calcul des TN par classe
        tn[cl_i] += np.sum((prf != cl_i) * (gtf != cl_i))
        #nombre de pixels par classe
        n_pixels[cl_i] += np.sum(gtf == cl_i)

    cl_IOU2 = (tp+0.000000000001) / (tp + fp + fn + 0.000000000001)
    cl_pct_pixel2 = n_pixels / np.sum(n_pixels)
    flat_IOU2 = np.sum(cl_IOU2*cl_pct_pixel2)
    mean_IOU2 = np.mean(cl_IOU2)
    
    #print("flat IOU sur le format original :","{:.2f}".format(flat_IOU2))
    #print("mean IOU sur le format original :","{:.2f}".format(mean_IOU2))
    
    #AFFICHAGE
    f, (b0, b1, b2) = plt.subplots(1, 3,figsize=(20,3), gridspec_kw={'width_ratios': [4, 4, 4]})

    b0.set_title('GT : Pourcentage de pixels par classe')
    b0.set_xticks(range(len(cats.keys())),cats.keys(),  rotation=45)
    b0.set_yticks(np.arange(0, max(cl_pct_pixel1)*100+10, 10))
    b0.bar(cats.keys(), cl_pct_pixel1*100, color=colors2)

    b1.set_title(('Prediction Mean IoU :{:.2f}'.format(mean_IOU1)+' format modèle {}'.format(model.output_width) +'X{}'.format(model.output_height)))
    b1.set_xticks(range(len(cats.keys())),cats.keys(),  rotation=45)
    b1.set_yticks(np.arange(0, 1, 0.1))
    b1.bar(cats.keys(), cl_IOU1, color=colors2)
    
    b2.set_title(('Prediction Mean IoU :{:.2f}'.format(mean_IOU2)+' format upsizé {}'.format(gt2c.shape[1]) +'X{}'.format(gt2c.shape[0])))
    b2.set_xticks(range(len(cats.keys())),cats.keys(),  rotation=45)
    b2.set_yticks(np.arange(0, 1, 0.1))
    b2.bar(cats.keys(), cl_IOU2, color=colors2)

    f.savefig('static/result_stat.png',bbox_inches='tight')
    #plt.show()
    
    


def model_from_checkpoint_path(config_path,checkpoints_path):



    model_config = json.loads(open(config_path+"_config.json", "r").read())
    

    model2load = vgg_unet(n_classes=model_config['n_classes'] ,  
                          input_height=model_config['input_height'], 
                          input_width=model_config['input_width'])
    
    #print("loaded weights ", latest_weights)
    status = model2load.load_weights(checkpoints_path)

    if status is not None:
        status.expect_partial()

    return model2load

#chargement du modèle
mod=model_from_checkpoint_path(config_path="model/vgg_unet_256_512/",checkpoints_path="model/vgg_unet_256_512/vgg_unet_4.00037")

'''dossier P8/flask/test/hello_color
un dossier static avec la feuille de style main.css
un dossier templates avec home.html

appel au css dans la balise link du template home.html qu'on peut ouvrir avec sublime text:
    on donne l adresse du css'
on intgere le message à envoyé grace à {{}} qui sera en type h1

dans main.css on specifie h1
    font size -> police 2em
    color -> green
    text-align -> center
    
CSS reference: https://www.w3schools.com/cssref/


Dans home.html
balise href permettant de mettre des liens
on lui donne le nom de la fonction et non la route

element form avec la methode post pour que l utilisateur puisse poster uen info
type text et bouton ok'''   

import os
    
#import du module Flask
from flask import Flask
#import du modèle render_template qui va faire le lien entre le programme python et la page html
from flask import render_template
#import du module request pour recuperer le message de l'utilisateur
from flask import request, redirect, url_for, flash
#to secure a filename
from werkzeug.utils import secure_filename

#upload folder
UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'png'}
SECRET_KEY = '12345' #necessaire pour faire marcher flash

#max 3Mo par file upload
MAX_CONTENT_PATH = 3000000

#fonction test d'extension eligible
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#nommage de l'application en app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_PATH'] = MAX_CONTENT_PATH
app.config['SECRET_KEY'] = SECRET_KEY

#on precise l'adresse de ce qui va suivre
#juste '/' signifie qu on 'est sur la page d accueil juste apre le nom de domaine'
@app.route('/')
#on appelle directement la fonction hello quand on arrive sur cette page
#on demande d'injecter le message dans le template de la page home
def hello():
    return render_template("index.html", modele_type =  "Pour cela, nous allons utilisé un modèle VGG_UNET 256x512 entrainé sur des images cityscapes")

#fonction permettant de recuperer les fichiers
@app.route('/', methods=['POST'])
def prediction():
    print(request.files.getlist("file[]"))
    files = request.files.getlist('file[]')
    if files ==[]:
        flash("Vous n'avez pas sélectionner de fichiers")
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
            flash('Fichier '+file.filename+' chargé')
            
    seg_prediction(model=mod,adress="static/im.png",gt_adress="static/gt.png")
    im2show=["result_image.png","result_stat.png"]
    return render_template("index.html", images=im2show)        
    #return redirect(url_for('hello'))

#@app.route('/', methods=['POST'])
#def prediction():
#    if "form-predict" in request.form:
#        seg_prediction(model=mod,adress="static/im.png",gt_adress="static/gt.png")
#        im2show=["result_image.png","result_stat.png"]
#        return render_template("home.html", images=im2show)
    

#ca lance l'application
if __name__ == "__main__":
    app.run()