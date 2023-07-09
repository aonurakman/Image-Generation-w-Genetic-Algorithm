#%%

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from tensorflow.keras.applications import InceptionResNetV2
from keras.preprocessing.image import img_to_array #, load_im
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as ppi_ires
from tensorflow.keras.applications.inception_resnet_v2 import decode_predictions

#%%

population_size = 100
generation = 100
mutation_rate = 0.01
cross = 1
img_shape = 299

plot_every = 50

#%%

copy_to_next = (int)(population_size - population_size / 3) # sonraki nesle direkt kopyalanan en iyi birey sayisi
#copy_to_next = 0
best_of_gen = np.zeros((generation), dtype=float) # her neslin en iyi bireyinin iyilik degerini tutar
mean_of_gen = best_of_gen.copy() # her neslin ortalama degerini tutar

max_f = 1.00

#%% Fitness Icın Kullanilacak Model

model = InceptionResNetV2(include_top=True, input_shape=(img_shape, img_shape, 3))
#model.summary()

#%% Ilk Generation

curr_gen = np.round(255*np.random.rand(population_size, img_shape, img_shape, 3)).astype(np.int)
img = curr_gen[0].copy()
plt.imshow(img)

#%% Fitness Skoru Metodu

def detector(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    img = ppi_ires(img)
    
    preds = model.predict(img)
    top_pred = decode_predictions(preds, top=1)[0][0][1:]
    return top_pred

#%% Genetik Algoritma
    
for i in range (0, generation):
    f_scores = np.zeros((population_size))
    
    f_labels = []
    
    for j in range (0, population_size):
        label, f_scores[j] = detector(curr_gen[j])
        f_labels.append(label)
        
    f_of_gen = f_scores / max_f
    n_f = f_of_gen / sum(f_of_gen)
    
    inds = np.argsort(n_f)
    rn_f = np.zeros(inds.size, dtype=int)
    cnt = 0
    for idx in inds:
        rn_f[idx] = cnt
        cnt += 1
    rn_f = rn_f / sum(rn_f)
    best_ind = np.argmax(rn_f)
    
    best_of_gen[i] = (float) (f_of_gen[best_ind])
    mean_of_gen[i] = (float) (np.mean(f_of_gen))
    
    if (i%plot_every == 0): # Plotting some generations
        print("\nGeneration #", str(i+1), str(f_scores[best_ind]), f_labels[best_ind])
        #plt.imshow(curr_gen[best_ind])
        im = Image.fromarray(curr_gen[best_ind].astype(np.uint8))
        im.save("Gen"+str(i+1)+"_"+str(f_scores[best_ind])+"_"+f_labels[best_ind]+".png")
        #plt.savefig("Gen"+str(i+1)+"_"+str(f_scores[best_ind])+"_"+f_labels[best_ind]+".png")

    if (i < generation-1): # eger son gen degilse cross ve mutasyon
        chosen = random.choices(range(0,population_size), weights = rn_f, k = population_size)
        # yeni bireyleri üret % tek/cift noktali crossover
        next_gen = np.zeros((population_size, img_shape, img_shape, 3), dtype=int) # yeni bireyler
        for j in range (0,(int)(population_size/2)):
            b = []
            b.append(curr_gen[chosen[j]])
            b.append(curr_gen[chosen[j+(int)(population_size/2)]])
            if cross == 1: # tek noktalı crossover
                cross_at = random.randint(1, img_shape-2) # 2 - (person_length-1) arasi sayi
                for idx in range (0, img_shape):
                    next_gen[j][idx][0:cross_at] = b[0][idx][0:cross_at] 
                    next_gen[j][idx][cross_at:] = b[1][idx][cross_at:]
                    next_gen[(int)(j+(population_size/2))][idx][0:cross_at] = b[1][idx][0:cross_at]
                    next_gen[(int)(j+(population_size/2))][idx][cross_at:] = b[0][idx][cross_at:]
            else:  # =2 noktalı crossover
                cross_at = [random.randint(1, img_shape-2)]
                cross_at.append(random.randint(1, img_shape-2))
                cross_at.sort() # kucukten buyuge sirala
                for idx in range (0, img_shape):
                    next_gen[j][idx][0:cross_at[0]] = b[0][idx][0:cross_at[0]]
                    next_gen[j][idx][cross_at[0]:cross_at[1]] = b[1][idx][cross_at[0]:cross_at[1]]
                    next_gen[j][idx][cross_at[1]:] = b[0][idx][cross_at[1]:]
                    next_gen[(int)(j+(population_size/2))][idx][0:cross_at[0]] = b[1][idx][0:cross_at[0]]
                    next_gen[(int)(j+(population_size/2))][idx][cross_at[0]:cross_at[1]] = b[0][idx][cross_at[0]:cross_at[1]]
                    next_gen[(int)(j+(population_size/2))][idx][cross_at[1]:] = b[1][idx][cross_at[1]:]

                
        if copy_to_next > 0: # current_gen deki en iyi copy_to_next degeri next_gen e kopyala
            for idx in inds[copy_to_next+1:]:
                next_gen[idx] = curr_gen[idx]
        
        # mutasyon uygula
        if mutation_rate > 0:
            for idx in range(0,population_size):
                cells_to_mutate = random.choices(range(0,(img_shape*img_shape)-1), k = (int)(((img_shape*img_shape)-1)*mutation_rate+1))
                new_values = random.choices(range(0,255), k = len(cells_to_mutate)) # nelerle degisecekleri
                for idx2 in range(0,len(cells_to_mutate)):
                    next_gen[idx][(int) (cells_to_mutate[idx2] / img_shape)][cells_to_mutate[idx2]%img_shape] = new_values[idx2]
        
        curr_gen = next_gen.copy() # yeni nesil hazir
        print("\n", i+1, ". GENERATION!")
    else:
        print("\nRESULT:")
        
    # Her nesilde en iyi bireyleri yazdir
    #print("Best individual:", current_gen[best_ind])
              
    print("\nBest score: ", best_of_gen[i], "\nMean score: ", mean_of_gen[i])
        
    
#%% Surec hakkinda veri gorsellestirmesi
    
im = Image.fromarray(curr_gen[best_ind].astype(np.uint8))
im.save("Gen"+str(i+1)+"_"+str(f_scores[best_ind])+"_"+f_labels[best_ind]+".png")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
