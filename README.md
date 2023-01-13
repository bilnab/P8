# P8  
<img src="/img/aml.png" width="150"> <img src="/img/keras.png" width="150"> <img src="/img/flask.png" width="150">     
 
**Projet deep learning de segmentation d'images**  
  
Dans le cadre de la concéption d'une voiture autonome, les sytèmes embarqués de vision s'articulent autour de 4 piliers
* l'acquisition d'image temps réél  
* le traitement d'images
* la segmentation des images que nous allons traité dans ce projet
* le système de décision

Ce projet a donc pour but d'entrainer et de déployer un modèle de segmentation d'images:  
* sur 8 catgéories principales
* en utilisant keras et azure ML
* via une API Flask


## Ressources:
images segmentées et annotées de caméras embarquées      
[lien 1](https://www.cityscapes-dataset.com/dataset-overview/)  
[lien 2](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_gtFine_trainvaltest.zip)    
[lien 3](https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/AI+Engineer/Project+8+-+Participez+%C3%A0+la+conception+d'une+voiture+autonome/P8_Cityscapes_leftImg8bit_trainvaltest.zip)            
   
Data generation et data augmentation:  
[Exemple de données cityscape et data generator](https://github.com/srihari-humbarwadi/cityscapes-segmentation-with-Unet/blob/master/batch_training.py)  
[Principes data generator](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly)  
[Tuto data generator](https://medium.com/datadriveninvestor/keras-training-on-large-datasets-3e9d9dbc09d4)  
[Data generator avec data augmentation](https://github.com/Golbstein/Keras-segmentation-deeplab-v3.1/blob/e3f0daaa79a729c022da658fc86eef82a6c7ceeb/utils.py#L411)  
[Lib d’augmentation d’images: albumentations](https://albumentations.ai/docs/examples/tensorflow-example/)  
[Lib d’augmentation d’images: imgaug](https://github.com/aleju/imgaug)  
Segmentation d’images:  	
[Principes segmentation d’images](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html)  
[Exemples](https://github.com/divamgupta/image-segmentation-keras)  
[Fonction loss pour la segmentation d’image](https://lars76.github.io/2018/09/27/loss-functions-for-segmentation.html)  
VRAC et AZURE:  
[Principes DNN et CNN](https://docs.microsoft.com/fr-fr/learn/modules/train-evaluate-deep-learn-models/)  
[Gestion donnéees azure](https://docs.microsoft.com/fr-fr/learn/modules/work-with-data-in-aml/)  
[Entrainement à distance azure](https://docs.microsoft.com/fr-fr/azure/machine-learning/tutorial-train-models-with-aml)  
[Authentification azure à partir de flask](https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/manage-azureml-service/authentication-in-azureml/authentication-in-azureml.ipynb)  
[Serialization d’images avec json](https://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions)  
[Azure webapp](https://docs.microsoft.com/fr-fr/azure/app-service/)  
[Creation et deployment azure webapp](https://docs.microsoft.com/fr-fr/learn/modules/host-a-web-app-with-azure-app-service/)  
[Mode de deployment de webapp](https://docs.microsoft.com/fr-fr/learn/modules/host-a-web-app-with-azure-app-service/6-deploying-code-to-app-service)  
[Creation de webapp sur le portail](https://docs.microsoft.com/fr-fr/learn/modules/host-a-web-app-with-azure-app-service/2-create-a-web-app-in-the-azure-portal)  
[Deploiement automatise de webapp via github](https://docs.microsoft.com/fr-fr/azure/app-service/deploy-continuous-deployment?tabs=github)  
[Creation et deployment de webapp en ligne de commande](https://docs.microsoft.com/fr-fr/azure/app-service/quickstart-python?tabs=bash&pivots=python-framework-flask)  
[Cours flask](https://openclassrooms.com/fr/courses/4425066-concevez-un-site-avec-flask)  


## Script   
[un notebook mettant en oeuvre la segmentation via SVC et random forest (pas terrible)](/P8_random%20forest%20&%20svc%20segmentation.ipynb)  
[un notebook mettant en oeuvre la segmentation deep learning](/P8_deep_segmentation.ipynb)  
  
 
 
## Présentation PDF:  
[pdf complet](/P8.pdf)  
<img src="/img/P8%20pres.png" height="300">   

## Etat de l'art  
[doc word](/P8%20segmentation_etat2lart.pdf)  

