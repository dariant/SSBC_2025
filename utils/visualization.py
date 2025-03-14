
from PIL import Image
import os 
import numpy as np
import matplotlib.pyplot as plt

def overlay_masks_on_samples(preds, inputs, labels, resume_where, ind):
    which_model = resume_where.split("/")[-1]

    trained_on = which_model.split("__")[0]

    save_to = "example_visualisations/"  + trained_on
    os.makedirs(save_to, exist_ok=True)
    
    preds = np.array(preds.cpu().detach().numpy())[0]
    inputs = np.array(inputs.cpu().detach().numpy())[0]
    labels = np.array(labels.cpu().detach().numpy())[0]

    inputs = (inputs * 255).astype(np.uint8).transpose(1, 2, 0)
    #inputs = Image.fromarray(inputs)

    # for multiclass use 25, 150, 150
    # for normal 4 class use 50, 60, 150 
    # old normal : 50, 50, 255... 255 
    factor = 50 #50 
    transparency = 60 #50 
    mask_factor = 150 # 255 #255 #150

    preds = (preds * factor).astype(np.uint8)
    labels = (labels * factor).astype(np.uint8)
    
    mask = Image.fromarray(preds, 'L') #.save('pred.png')
    og_img = Image.fromarray(inputs)#.save('input.png')

    # Get the color map by name:
    cm = plt.get_cmap("gist_ncar")#plt.get_cmap("jet") # pcik cm #"jet" # TODO

    img = og_img.copy()
    mask_transparent = np.array(cm(mask))[:, :, :3]
    mask_transparent = Image.fromarray((mask_transparent * mask_factor).astype(np.uint8)).convert("RGBA")  # *10 withoutbefore TODO

    mask_transparent.putalpha(transparency)
    img.paste(mask_transparent, (0,0), mask_transparent)
    img.save(save_to + "/" + str(ind) + "_" + which_model  + ".png")
    
    
    # GROUND TRUTHS TODO save
    gt = Image.fromarray(labels, 'L') #.save('truth.png')
    mask_transparent = np.array(cm(gt))[:, :, :3]
    mask_transparent = Image.fromarray((mask_transparent * mask_factor).astype(np.uint8)).convert("RGBA") #* 10 # was * 255

    mask_transparent.putalpha(transparency)
    og_img.paste(mask_transparent, (0,0), mask_transparent)
    og_img.save(save_to + "/" + str(ind) + "_GT_" + which_model  + ".png")
    
    og_img.save(save_to + "/example_GT_merge_" + which_model + "_" +str(ind) + ".png")

    return 


def visualize_predicted_mask(preds, inputs, labels, resume_where, ind):
    which_model = resume_where.split("/")[-1]

    trained_on = which_model.split("__")[0]

    save_to = "example_visualisations/"  + trained_on
    os.makedirs(save_to, exist_ok=True)
    
    preds = np.array(preds.cpu().detach().numpy())[0]
    inputs = np.array(inputs.cpu().detach().numpy())[0]
    labels = np.array(labels.cpu().detach().numpy())[0]

    inputs = (inputs * 255).astype(np.uint8).transpose(1, 2, 0)
    #inputs = Image.fromarray(inputs)

    # for multiclass use 25, 150, 150
    # for normal 4 class use 50, 60, 150 
    # old normal : 50, 50, 255... 255 
    factor = 50 #50 
    transparency = 60 #50 
    mask_factor = 150 # 255 #255 #150

    preds = (preds * factor).astype(np.uint8)
    labels = (labels * factor).astype(np.uint8)
    
    mask = Image.fromarray(preds, 'L') #.save('pred.png')
    og_img = Image.fromarray(inputs)#.save('input.png')

    # Get the color map by name:
    cm = plt.get_cmap("gist_ncar")#plt.get_cmap("jet") # pcik cm #"jet" # TODO

    img = og_img.copy()
    mask_transparent = np.array(cm(mask))[:, :, :3]
    mask_transparent = Image.fromarray((mask_transparent * mask_factor).astype(np.uint8)).convert("RGBA")  # *10 withoutbefore TODO

    mask_transparent.putalpha(transparency)
    img.paste(mask_transparent, (0,0), mask_transparent)
    img.save(save_to + "/" + str(ind) + "_" + which_model  + ".png")
    
    
    # GROUND TRUTHS TODO save
    gt = Image.fromarray(labels, 'L') #.save('truth.png')
    mask_transparent = np.array(cm(gt))[:, :, :3]
    mask_transparent = Image.fromarray((mask_transparent * mask_factor).astype(np.uint8)).convert("RGBA") #* 10 # was * 255

    mask_transparent.putalpha(transparency)
    og_img.paste(mask_transparent, (0,0), mask_transparent)
    og_img.save(save_to + "/" + str(ind) + "_GT_" + which_model  + ".png")
    
    og_img.save(save_to + "/example_GT_merge_" + which_model + "_" +str(ind) + ".png")

    return 