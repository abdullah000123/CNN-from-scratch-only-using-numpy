import numpy as np
import os as s
import cv2 as cv
import matplotlib.pyplot as plt
#print(np.cuda.is_available())
print('loading data')
data='data/train'
labels=[]
images=[]
image_size=120
for imagefile in s.listdir(data):
    image_path = s.path.join(data,imagefile)
    image=cv.imread(image_path,cv.IMREAD_COLOR)
    if image is None:
        print("Failed to load image.")
    image= cv.resize(image,(image_size,image_size))
    if imagefile.startswith("c"):
        label=0
    elif imagefile.startswith("d"):
        
        label=1
    labels.append(label)
    images.append(image)
print('labels',len(labels))
print('images',len(images))

labels=np.array(labels)
images=np.array(images)
#device=np.device('cuda:0')
print('label',labels[0])
#print('image',images[0])
print('ARRAY images dimention',images.shape)

print('image dimention',images[0].shape)
#print('image 2',images[1])
#print('image  2 dimention ',images[1].shape)
#now cnn
stride=1
padding=0


batch_sz=1

num_batches=len(images)//batch_sz
batch_img=[]
batch_lab=[]
for h in range(num_batches):
    start=h*batch_sz
    end=(h+1)*batch_sz
    batch=images[start:end]
    batch_l=labels[start:end]
    batch_img.append(batch)
    batch_lab.append(batch_l)
batch_img=np.array(batch_img)
batch_lab=np.array(batch_lab)
print(batch_img[0].shape)
print(batch_lab[0])
#plt.imshow(batch_img[0])
print('label',batch_lab[0])




def relu(feature_map):
    relu_out = np.zeros(feature_map.shape)
    for map_num in range(feature_map.shape[0]):
        for r in np.arange(0,feature_map.shape[1]):
            for c in np.arange(0, feature_map.shape[-1]):
                relu_out[ map_num,r, c] = np.max([feature_map[ map_num,r, c], 0])
    return relu_out
    
def relu_derivative(da,z):
    return da*(z>0)

def conv(l1_filter,images):
    image_height, image_width, image_channels = image.shape
    
    # If padding is applied, pad the image
    if padding > 0:
        images = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    # Compute the output dimensions
    output_height = (image_height - l1_filter[0].shape[0]) // stride + 1
    output_width = (image_width - l1_filter[0].shape[1]) // stride + 1
    
    output = np.zeros((output_height, output_width,3))
    feature_map=[]
    for k in range(l1_filter.shape[0]):
        filter_=l1_filter[k]
        for c in range(3):
            for i in range(0, output_height):
                for j in range(0, output_width):
                    region = image[i:i+filter_.shape[0], j:j+filter_.shape[1],c]
                    output[i,j, c] = np.sum(region * filter_[:,:])
        output1=np.sum(output,axis=2)
        feature_map.append(output1)
    as1_l=np.array(feature_map)
    as1=relu(as1_l)
    return as1,as1_l
    
def conv2(l1_filter,images):
    image_channels , image_height , image_width= images.shape
    no_of_f ,  f_size_h , f_size_w=l1_filter.shape
    # If padding is applied, pad the image
    if padding > 0:
        images = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    # Compute the output dimensions
    output_height = (image_height - f_size_h) // stride + 1
    output_width = (image_width - f_size_w) // stride + 1
    
    # Initialize the output feature map
    output_of_layer = np.zeros((no_of_f,output_height, output_width))
    feature_map=[]
    for k in range(no_of_f):
        for h in range(output_height):
            for w in range(0, output_width):
                output_of_layer[k,h,w] = np.sum(images[: ,h:h+f_size_h, w:w+f_size_w] * l1_filter[k])
                
    output=relu(output_of_layer)
    return output,output_of_layer

def conv_backward(dZ, weight, input_maps,layer, stride=1, padding=0):
    A_prev = input_maps
    (pre_map, n_H_prev, n_W_prev) = A_prev.shape
    (n_C, f, f) = weight.shape
    (m,nh,nw)=dZ.shape
    if layer==1:
        pre_map=3
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(weight)
    db = np.zeros((n_C, 1))

    for c in range(pre_map): 
        a_prev=A_prev[c]
        for h in range(nh):               
            for w in range(nw):           
                vert_start = h*stride 
                vert_end = vert_start + f
                horiz_start = w*stride
                horiz_end = horiz_start + f
                        # Use the corners to define the slice from a_prev_pad
                if layer!=1:
                        # gradient of sum (f*a_slide) = sum of gradients (f*dZ)
                    a_slice = A_prev[c,vert_start: vert_end, horiz_start:horiz_end]
                    dA_prev[:,vert_start:vert_end, horiz_start:horiz_end] += (weight[c,:,:] * dZ[c,h,w]).astype(np.uint8)
                else:
                    a_slice = A_prev[vert_start: vert_end, horiz_start:horiz_end,:]
                    a_slice =np.sum(a_slice,axis=2)
                    #dA_prev[vert_start:vert_end, horiz_start:horiz_end,:] += (weight[c,:,:] * dZ[c,h,w]).astype(np.uint8)

                dW[c,:,:] += a_slice * dZ[c,h,w]
    return dA_prev, dW, db


def pooling_max(s, images, ps):
    maps, height, width = images.shape
    pool_height, pool_width = ps, ps
    output_height = (height - pool_height) // s + 1
    output_width = (width - pool_width) // s + 1
    pooling_out = np.zeros((maps, output_height, output_width))
    global cache_p

    max_positions_row = np.zeros((maps, output_height, output_width), dtype=int)
    max_positions_col = np.zeros((maps, output_height, output_width), dtype=int)
    #max_p=np.zeros((maps, output_height, output_width), dtype=int)
    for i in range(maps):
        for h in range(output_height):
            for w in range(output_width):
                vert_start = h * s
                vert_end = vert_start + pool_height
                horiz_start = w * s
                horiz_end = horiz_start + pool_width
                pool_region = images[i, vert_start:vert_end, horiz_start:horiz_end]
                # Find the max value and its index in the pool region
                max_val = np.max(pool_region)
                max_row, max_col = np.unravel_index(pool_region.argmax(), pool_region.shape)
                # Store the max value and its indices
                pooling_out[i, h, w] = max_val
                max_positions_row[i, h, w] = vert_start + max_row
                max_positions_col[i, h, w] = horiz_start + max_col
                #max_p[i, h, w]=np.unravel_index(pool_region.argmax(), pool_region.shape)
    # Cache for backpropagation
    cache_p = (images, max_positions_row, max_positions_col)
    #cache_p=(images,max_p)
    return pooling_out, cache_p

def pooling_avg(s,images,ps):
    pooling_out=np.zeros(images.shape)
    pooling_size=ps
    size_height=(images.shape[1]-pooling_size)//s +1
    size_width=(images.shape[2]-pooling_size)//s +1
    pooling_out=np.zeros((images.shape[0],size_height,size_width))
    for maps in range(0,images.shape[0]):
        image=images[maps]
        for i in range(0, images.shape[1]-pooling_size):
            for j in range(0, images.shape[2]-pooling_size):
                if s==1:
                    region = image[i:i+pooling_size, j:j+pooling_size]
                    pooling_out[maps,i,j]=np.average(region)
                else:
                    region = image[i+s:i+pooling_size+s, j+s:s+j+pooling_size]
                    pooling_out[maps,i,j]=np.average(region)
                        
    return pooling_out

def pooling_min(s,images,ps):
    maps, height, width = images.shape
    pool_height, pool_width = ps, ps
    output_height = (height - pool_height) // s + 1
    output_width = (width - pool_width) // s + 1
    pooling_out = np.zeros((maps, output_height, output_width))
    global cache_p

    max_positions_row = np.zeros((maps, output_height, output_width), dtype=int)
    max_positions_col = np.zeros((maps, output_height, output_width), dtype=int)
    #max_p=np.zeros((maps, output_height, output_width), dtype=int)
    for i in range(maps):
        for h in range(output_height):
            for w in range(output_width):
                vert_start = h * s
                vert_end = vert_start + pool_height
                horiz_start = w * s
                horiz_end = horiz_start + pool_width
                pool_region = images[i, vert_start:vert_end, horiz_start:horiz_end]
                # Find the max value and its index in the pool region
                max_val = np.min(pool_region)
                max_row, max_col = np.unravel_index(pool_region.argmin(), pool_region.shape)
                # Store the max value and its indices
                pooling_out[i, h, w] = max_val
                max_positions_row[i, h, w] = vert_start + max_row
                max_positions_col[i, h, w] = horiz_start + max_col
                #max_p[i, h, w]=np.unravel_index(pool_region.argmax(), pool_region.shape)
    # Cache for backpropagation
    cache_p = (images, max_positions_row, max_positions_col)
    #cache_p=(images,max_p)
    return pooling_out, cache_p
   # stride
def pooling_max_backward(dA, cache, pool_size):
    (A_prev, max_positions_row,max_positions_col) = cache
    (m, h_prev, w_prev) = A_prev.shape
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):  
        for h in range(dA.shape[1]):  
            for w in range(dA.shape[2]):  
                # Retrieve the corresponding max positions
                vert_start = h * stride
                horiz_start = w * stride

                max_row = max_positions_row[i, h, w]
                max_col = max_positions_col[i, h, w]

                dA_prev[i, max_row, max_col] += dA[i, h, w]
    #print('reverse pooling done',dA_prev.shape)
    return dA_prev
def flatten_forward(x):
    # Store the original shape to reshape during backpropagation
    original_shape = x.shape
    #print('orignal shape before  forward flatten', original_shape)
    flattened = x.reshape(1, -1)  # Flatten to (batch_size, height * width * channels)
    return flattened, original_shape
def flatten_backward(d_flattened, original_shape):
    #print('result  ',d_flattened.reshape(original_shape).shape)

    return d_flattened.reshape(original_shape)

no_of_filters=3
s=1  
p=0
l1_filter=np.random.randn(32,3,3)*0.1
l2_filter=np.random.randn(64,4,4)*0.1
l3_filter=np.random.randn(64,5,5)*0.1



# Activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Forward propagation
def forward_propagation_dense(x):
    global a1, a2, a3
    z11 = np.dot(x, theta0) + bias0
    a1 = (z11)
    #print('dense layer 1 done')

    z22 = np.dot(a1, theta1) + bias1
    a2 = sigmoid(z22)
    #print('dense layer 2 done')

    z33 = np.dot(a2, theta2) + bias2
    a3 = sigmoid(z33)
    #print('dense layer 3 done')
    #print('a3  final',a3.shape)
    return a3

def back_propagation_dense(x, y, a3):
    global a1, a2
    error = a3 - y
    dZ3 = error * sigmoid_derivative(a3)
    dA2 = np.dot(dZ3, theta2.T) * sigmoid_derivative(a2)
    dZ2 = dA2
    dA1 = np.dot(dZ2, theta1.T) * sigmoid_derivative(a1)
    dA0 = np.dot(dA1, theta0.T) 


    dtheta2 = np.dot(a2.T, dZ3)
    dbias2 = np.sum(dZ3, axis=0, keepdims=True)

    dtheta1 = np.dot(a1.T, dZ2)
    dbias1 = np.sum(dZ2, axis=0, keepdims=True)

    dtheta0 = np.dot(x.T, dA1)
    dbias0 = np.sum(dA1, axis=0, keepdims=True)

    return dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2,dA0
    
def cnn_forward(image):
    #print('starting training')
    global cache_1,cache_2,cache_3,out,out2,out3,z1,z2,z3

    output ,  z1 =conv(l1_filter,image)
    
    out,cache_1=pooling_max(s,output,3)
    #print('conv 1 and pooling done')
    result_l2,z2=conv2(l2_filter,out)
    
    out2,cache_2=pooling_max(s,result_l2,4)
    #print('conv 2 and pooling done')

    result_l3,z3=conv2(l3_filter,out2)
    
    out3,cache_3=pooling_max(s,result_l3,4)
    #print('conv 3 and pooling done')

    plt.imshow(out3[1])
    
    global original_shape
    final, original_shape = flatten_forward(out3)
    #print("Flattened shape:", final.shape)  #  (32, 28*28*3)
    return final,original_shape
    

def cnn_backpropogation(x):
    x1=flatten_backward(x,original_shape)
    x2=pooling_max_backward(x1,cache_3,2)
    x3=relu_derivative(x2,z3)
    l3_c,dw3,db3=conv_backward(x3,l3_filter,out2,3)
    
    x4=pooling_max_backward(l3_c,cache_2,2)
    x5=relu_derivative(x4,z2)
    l2_c,dw2,db2=conv_backward(x5,l2_filter,out,2)

    x6=pooling_max_backward(l2_c,cache_1,2)
    x7=relu_derivative(x6,z1)
    l1_c,dw1,db1=conv_backward(x7,l1_filter,image,1)
    
    return dw3,dw2,dw1,db3,db2,db1
    
def trainf( lr, epochs,label):
    global theta1, theta2, bias1, bias2, theta0, bias0, l1_filter, l2_filter, l3_filter
    total_loss = 0
    final , orignal_out=cnn_forward(batch_img[0])
         # Initialization of weights and biases
    no_input = final.shape[1]   
    no_l2 = 128       # Hidden layer 1
    no_l3 = 28       # Hidden layer 2
    no_out = 2     # Output layer for 10 classes
        
     
    theta0 = np.random.rand(no_input, no_l2) * 0.001
    bias0 = np.zeros((1, no_l2))
    theta1 = np.random.rand(no_l2, no_l3) * 0.001
    bias1 = np.zeros((1, no_l3))
    theta2 = np.random.rand(no_l3, no_out) * 0.001
    bias2 = np.zeros((1, no_out))
    for i in range(epochs):
        total=0
        for b in range(batch_img.shape[0]):
            images=batch_img[b]
            label=batch_lab[b]
            final , orignal_out=cnn_forward(images)
       
            inputs = final.reshape(-1, no_input)
            #print('flatten to dns',inputs.shape)
        
            outputs = forward_propagation_dense(inputs)
            #print('final out for ffn is ',outputs.shape)
            #calculate the binary cross-entropy loss
            loss = -(label * np.log(outputs) + (1 - label) * np.log(1 - outputs)).mean()
            total+=loss
                # Backpropagation
            dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2,DA1 = back_propagation_dense(inputs, label, outputs)
            
             # Update Dense Layer weights and biases
            theta0 -= lr * dtheta0
            bias0 -= lr * dbias0
            theta1 -= lr * dtheta1
            bias1 -= lr * dbias1
            theta2 -= lr * dtheta2
            bias2 -= lr * dbias2
        
            dw3,dw2,dw1,db3,db2,db1=cnn_backpropogation(DA1)
            # Update filters for Convolutional Layer 
            l3_filter += lr * dw3
            l2_filter += lr * dw2
            l1_filter += lr * dw1
                # Print average loss per epoch
            print(f"batch , Loss" ,b+1,loss)
        print(f"epoch , Loss" ,i+1,total/batch_img[0].shape)

lr = 0.001
epochs = 10
label = labels[0] 
trainf(lr, epochs ,label)


         
           
       
