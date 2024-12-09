import numpy as np
import os as s
import cv2 as cv
import matplotlib.pyplot as plt

print('loading data')
data = 'data/shapes'  # Path to the dataset
labels = []
images = []
image_size = 60

# Loop through the dataset folder
for imagefile in s.listdir(data):
    image_path = s.path.join(data, imagefile)
    image = cv.imread(image_path, cv.IMREAD_COLOR)  # Read image
    if image is None:
        print(f"Failed to load image: {imagefile}")
        continue  # Skip if image fails to load

    image = cv.resize(image, (image_size, image_size)) # Resize image to (120, 120)

    # Assign labels based on file name
    if imagefile.startswith("C"):
        label = 0
    elif imagefile.startswith("T"):
        label = 1
    else:
        label = 2
    labels.append(label)
    images.append(image)

print('Labels count:', len(labels))
print('Images count:', len(images))

# Convert lists to NumPy arrays
labels = np.array(labels)
images = np.array(images)

# Convert labels to one-hot encoding
num_classes = 3  # Number of classes
labels_one_hot = np.eye(num_classes)[labels]

# Shuffle  data while preserving  pairing between images and labels
print("Shuffling data...")
indices = np.random.permutation(len(images))  # Generate random indices
images = images[indices]  # Shuffle images
labels_one_hot = labels_one_hot[indices]  # Shuffle corresponding one-hot encoded labels

# Print shuffled data details
print('Sample label (one-hot encoded):', labels_one_hot[0])
print('Array images dimension:', images.shape)
print('Image dimension:', images[0].shape)
print('Labels one-hot shape:', labels_one_hot.shape)



#now cnn
stride = 1
padding = 0


batch_sz = 60
num_batches = len(images) // batch_sz
batch_img = []
batch_lab = []

for h in range(num_batches):
    start = h * batch_sz
    end = (h + 1) * batch_sz
    batch = images[start:end]
    batch_l = labels_one_hot[start:end]
    batch_img.append(batch)
    batch_lab.append(batch_l)
batch_img = np.array(batch_img)
batch_lab = np.array(batch_lab).astype(np.int8)
# Activation function 
def relu(x):
    return np.maximum(0, x)
    
def relu_derivatived(x):
    return (x > 0)
        
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / (np.sum(exp_x, axis=1, keepdims=True))

def relu_derivative(da,z):
    return da * (z>0)


def conv(l1_filter,images,b):
    image_height, image_width, image_channels = images.shape
    
    # If padding is applied, pad the image
    if padding > 0:
        images = np.pad(images, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    stride = 1
    # Compute the output dimensions
    output_height = (image_height - l1_filter[0].shape[0]) // stride + 1
    output_width = (image_width - l1_filter[0].shape[1]) // stride + 1
    output = np.zeros((l1_filter.shape[0],output_height, output_width,3))
    for k in range(l1_filter.shape[0]):
        for c in range(3):
            for i in range(0, output_height):
                for j in range(0, output_width):
                    #region = region = images[i:i+l1_filter[0].shape[0], j:j+l1_filter[0].shape[1],c]
                    output[k ,i ,j ,c] = np.sum(images[i:i+l1_filter[0].shape[0], j:j+l1_filter[0].shape[1] ,c] * l1_filter[k])
        output1 =np.sum(output ,axis=3)
    as1 = relu(output1 + b)
    return as1 ,(output1 + b)


    
def conv2(l1_filter ,images , b):
    image_channels , image_height , image_width = images.shape
    no_of_f ,  f_size_h , f_size_w = l1_filter.shape
    # If padding is applied, pad the image
    if padding > 0:
        images = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode = 'constant')
    stride = 1
    # Compute the output dimensions
    output_height = (image_height - f_size_h) // stride + 1
    output_width = (image_width - f_size_w) // stride + 1
    
    # Initialize the output feature map
    output_of_layer = np.zeros((no_of_f ,output_height , output_width))
    for k in range(no_of_f):
        for h in range(output_height):
            for w in range(0, output_width):
                output_of_layer[k ,h ,w] = np.sum(images[: ,h:h+f_size_h , w:w+f_size_w] * l1_filter[k])
    output=relu(output_of_layer  + b)  
    return output ,(output_of_layer + b)

def conv_backward(dZ, weight, input_maps ,layer , padding=0):
    A_prev = input_maps       # input to layer 
    #(pre_map, n_H_prev, n_W_prev) = A_prev.shape
    (no_filters, f, f) = weight.shape #filter of current layer
    (m,nh,nw) = dZ.shape       #derivative of loss wrt ouput
    #print('dz and input to this layer in back propogation maps',dZ.shape[0],A_prev.shape[0])
    #print('no of filter',weights.shape[0])
    if layer == 1 :
        pre_map = 3                                                                                                                #changed
        n_H_prev = A_prev[0]
        n_W_prev = A_prev[1]
        stride = 1
    else:
        (pre_map, n_H_prev, n_W_prev) = A_prev.shape
        stride = 1
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(weight)
    db = np.zeros((no_filters, 1,1))
    for c in range(no_filters): 
        for h in range(nh) :               
            for w in range(nw) : 
                
                vert_start = h * stride 
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f
         
                if layer!=1:
                            # gradient of sum (f*a_slide) = sum of gradients (f*dZ)
                    a_slice = A_prev[: ,vert_start: vert_end, horiz_start:horiz_end]
                    a_slice =np.sum(a_slice,axis=0)

                    dA_prev[:,vert_start:vert_end, horiz_start:horiz_end] += (weight[c,:,:] * dZ[c,h,w])#.astype(np.float32)
                else:
                    a_slice = A_prev[vert_start: vert_end, horiz_start:horiz_end ,:]                      #changed :
                    a_slice = np.sum(a_slice,axis=2)
                     
                dW[c,:,:] += a_slice * dZ[c,h,w]
                db[c,:,:] += dZ[c, h, w]
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


def pooling_backward(dA, cache, pool_size):
    (A_prev, positions_row,positions_col) = cache
    (m, h_prev, w_prev) = A_prev.shape
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):  
        for h in range(dA.shape[1]):  
            for w in range(dA.shape[2]):  
                # Retrieve the corresponding max positions
                vert_start = h * stride
                horiz_start = w * stride

                max_row = positions_row[i, h, w]
                max_col = positions_col[i, h, w]

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


# Forward propagation
def forward_propagation_dense(x):
    global a1, a2, a3 ,a4
    z11 = np.dot(x, theta0) + bias0
    a1 = relu(z11)
    #print('a1',a1.shape)

    z22 = np.dot(a1, theta1) + bias1
    a2 = relu(z22)
    #print('a2',a2.shape)

    z33 = np.dot(a2, theta2) + bias2
    a3 = relu(z33)
    
    z44= np.dot(a3, theta3) + bias3
    #print('input to softmax ',z44)
    a4 = softmax(z44)
    
    #a3=np.argmax(a3)
    #print('dense layer 3 done')
    #print('a3  final',a3.shape)
    return a4

    
def back_propagation_dense(x, y, a4):
    global ls , dZ4 ,dZ3  ,dZ2  ,dA1 ,true_label
    true_label = y
    ls = a4 - true_label
    #print('a3',a3.shape, a3)
    #print('y',y.shape, y)
    dZ4 = ls  
    #print('dZ3',dZ3)
    dZ3 = np.dot(dZ4, theta3.T) * relu_derivatived(a3)

    dZ2 = np.dot(dZ3, theta2.T) * relu_derivatived(a2)
    #print('dZ2',dZ2.shape)

    dA1 = np.dot(dZ2, theta1.T) * relu_derivatived(a1)
    #print('dA1',dA1.shape)
    global dA0

    dA0 = np.dot(dA1, theta0.T) 
    #print('dA0',dA0.shape)
    
    dtheta3 = np.dot(a3.T, dZ4)
    dbias3 = np.sum(dZ4, axis=0, keepdims=True)
    
    dtheta2 = np.dot(a2.T, dZ3)
    dbias2 = np.sum(dZ3, axis=0, keepdims=True)

    dtheta1 = np.dot(a1.T, dZ2)
    dbias1 = np.sum(dZ2, axis=0, keepdims=True)

    dtheta0 = np.dot(x.T, dA1)
    dbias0 = np.sum(dA1, axis=0, keepdims=True)
    
    return dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2, dtheta3, dbias3
    
   
def cnn_forward(image):
    global cache_1 ,cache_2 ,cache_3 ,cache_4 ,cache_5 , out ,out2 ,out3  ,z1 ,z2 ,z3 ,z4 ,z5 ,result_l4 ,result_l3
    
    output ,  z1 =conv(l1_filter ,image ,b1)
    out ,cache_1 = pooling_max(s ,output ,pooling_size)
    #print('conv 1 and pooling done')
    #print('c and max shape 1',out.shape)
    result_l2 ,z2 = conv2(l2_filter ,out ,b2)
    out2 ,cache_2 = pooling_max(s ,result_l2 ,pooling_size)   
    #print('conv 2 and pooling done'
    #print('c and max shape 2',out2.shape)

    result_l3 ,z3 = conv2(l3_filter ,out2 ,b3)
    #print('conv 3 and pooling done')
    #print('c 3',result_l3.shape)
    result_l4 ,z4 = conv2(l4_filter ,result_l3 ,b4)
    #print('c 4',result_l4.shape)

    result_l5 ,z5 = conv2(l5_filter ,result_l4 ,b5)
    out5 ,cache_5 = pooling_max(s, result_l5 ,pooling_size)
    #print('conv 3 and pooling done')
    #print('c and max shape 5',out5.shape)
    global original_shape
    final, original_shape = flatten_forward(out5)
    #print("Flattened shape:", final.shape)  
    return final ,original_shape
    

def cnn_backpropogation():
    x = flatten_backward(dA0 ,original_shape)
    #print('c and max shape 5 gradient ',x.shape)
    back = pooling_backward(x , cache_5 , pooling_size)
    z_back = relu_derivative(back , z5)
    l5_c, dw5 ,db5 =conv_backward(z_back , l5_filter , result_l4 , 5)
    
    #print('c 4 back',l5_c.shape)

    z_back2 = relu_derivative(l5_c , z4)
    l4_c, dw4 ,db4 =conv_backward(z_back2 , l4_filter , result_l3 , 4)
    #print('c 3 back',l4_c.shape)

    x3 = relu_derivative(l4_c , z3)
    l3_c,dw3,db3 = conv_backward(x3 , l3_filter , out2 , 3)
    #print('c and max shape 2',l3_c.shape)

    x4 = pooling_backward(l3_c , cache_2 , pooling_size)
    x5 = relu_derivative(x4 ,z2)
    l2_c ,dw2 ,db2 = conv_backward(x5 , l2_filter , out , 2)
    #print('c and max shape 1',l2_c.shape)

    x6 = pooling_backward(l2_c , cache_1 , pooling_size)
    x7 = relu_derivative(x6 ,z1)
    l1_c ,dw1 ,db1 = conv_backward(x7 , l1_filter , image , 1)
    #print('input img in back prop and backprop done',l1_c.shape)

    return dw5 ,dw4 ,dw3 ,dw2 ,dw1 ,db5 ,db4 ,db3 ,db2 ,db1


def categorical_cross_entropy_loss(y_pred, y_true):

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Compute the categorical cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    # Return the loss
    return loss
 
def trainf( lr, epochs):
    global theta1, theta2, bias1, bias2, theta0, bias0, theta3, bias3, l1_filter, l2_filter, l3_filter ,l4_filter ,l5_filter ,b1 ,b2 ,b3 ,b4 ,b5 , s ,p ,pooling_size
    s = 2  
    p = 0
    
    pooling_size = 3

    l1_filter = np.random.randn(96,5,5)*0.01
    #l1_filter = np.abs(l1_filter)
    b1 = np.zeros((96,1, 1))
    
    l2_filter = np.random.randn(256,5,5)*0.01
    #l2_filter = np.abs(l2_filter)
    b2 = np.zeros((256,1, 1))
    
    l3_filter = np.random.randn(384,3,3)*0.01
    #l3_filter = np.abs(l3_filter)
    b3 = np.zeros((384,1, 1))
    
    l4_filter = np.random.randn(384,3,3)*0.01
    #l4_filter = np.abs(l4_filter)
    b4 = np.zeros((384,1, 1))
    
    l5_filter = np.random.randn(256,3,3)*0.01
    #l4_filter = np.abs(l4_filter)
    b5 = np.zeros((256,1, 1))
    
    total_loss = 0
    
    batch_imgsss = batch_img[0]
    print('starting first ')
    final , orignal_out = cnn_forward(batch_imgsss[0])
             # Initialization of weights and biases
    #print('final shape',final.shape)
    no_input = final.shape[1]   
    no_l2 = 4096     # Hidden layer 1
    no_l3 = 4096
    no_l4 = 256     # Hidden layer 2
    no_out = 3      # Output layer for 2 classes
    
    theta0 = np.abs( np.random.rand(no_input, no_l2) * 0.01 )
    bias0 = np.zeros((1, no_l2))
    
    theta1 = np.abs( np.random.rand(no_l2, no_l3) * 0.01 )
    bias1 = np.zeros((1, no_l3))
    
    theta2 = np.abs(  np.random.rand(no_l3, no_l4) * 0.01 )
    bias2 = np.zeros((1, no_l4))
    
    theta3 = np.abs( np.random.rand(no_l4, no_out) * 0.01 )
    bias3 = np.zeros((1, no_out))
    
    inputs = final.reshape(-1, no_input)
                #print('flatten to dns',inputs.shape)
            
    outputs = forward_propagation_dense(inputs)
    dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2 ,dtheta3, dbias3 = back_propagation_dense(inputs , batch_lab[0,0] , outputs )
    print('back prop dense done ')
    
    dw5 ,dw4 ,dw3 ,dw2 ,dw1 ,db5 ,db4 ,db3 ,db2 ,db1 = cnn_backpropogation()


    for i in range(epochs):
        print('epoch no ',i + 1)
        total_lossst = 0
        for ba in range(num_batches):
            batch_im = batch_img[ba]
            batch_ll = batch_lab[ba]
            total_loss = 0
            batch_dtheta0 = np.zeros_like(dtheta0)
            batch_dbias0 = np.zeros_like(dbias0)
            batch_dtheta1 = np.zeros_like(dtheta1)
            batch_dbias1 = np.zeros_like(dbias1)
            batch_dtheta2 = np.zeros_like(dtheta2)
            batch_dbias2 = np.zeros_like(dbias2)
            batch_dtheta3 = np.zeros_like(dtheta3)
            batch_dbias3 = np.zeros_like(dbias3)
            
            batch_dw5 = np.zeros_like(dw5)
            batch_dw4 = np.zeros_like(dw4)
            batch_dw3 = np.zeros_like(dw3)
            batch_dw2 = np.zeros_like(dw2)
            batch_dw1 = np.zeros_like(dw1)

            batch_db5 = np.zeros_like(db5)
            batch_db4 = np.zeros_like(db4)
            batch_db3 = np.zeros_like(db3)
            batch_db2 = np.zeros_like(db2)
            batch_db1 = np.zeros_like(db1)
            
            print('\nStarting batch => ',ba+1)
            
            for im in range(batch_sz):
                final , orignal_out = cnn_forward(batch_im[im])
           
                inputs = final.reshape(-1, no_input)
                #print('flatten to dns',inputs.shape)
            
                outputs = forward_propagation_dense(inputs)
                #print('\nActual is => ', batch_ll[im] )
                #print('Predicted is => ',outputs)
                # binary cross-entropy loss
                loss = categorical_cross_entropy_loss(outputs, batch_ll[im])               
                #print('#######loss###### =>',loss)
                total_loss += loss
                    # Backpropagation
                dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2 ,dtheta3, dbias3 = back_propagation_dense(inputs, batch_ll[im] , outputs )
                
                batch_dtheta0 += dtheta0
                batch_dbias0 += dbias0
                
                batch_dtheta1 += dtheta1
                batch_dbias1 += dbias1
                
                batch_dtheta2 += dtheta2
                batch_dbias2 += dbias2
                
                batch_dtheta3 += dtheta3
                batch_dbias3 += dbias3
                
                dw5 ,dw4 ,dw3 ,dw2 ,dw1 ,db5 ,db4 ,db3 ,db2 ,db1 = cnn_backpropogation()
                
                batch_dw5 += dw5
                batch_dw4 += dw4
                batch_dw3 += dw3
                batch_dw2 += dw2
                batch_dw1 += dw1
                
                batch_db5 += db5
                batch_db4 += db4
                batch_db3 += db3
                batch_db2 += db2
                batch_db1 += db1
                
                 # Update Dense Layer weights and biases
            theta0 -= lr * batch_dtheta0 / batch_sz
            bias0 -= lr * batch_dbias0 / batch_sz
            theta1 -= lr * batch_dtheta1 / batch_sz
            bias1 -= lr * batch_dbias1 / batch_sz
            theta2 -= lr * batch_dtheta2 / batch_sz
            bias2 -= lr * batch_dbias2 /  batch_sz
            theta3 -= lr * batch_dtheta3 / batch_sz
            bias3 -= lr * batch_dbias3 / batch_sz
            
            
                # Update filters for Convolutional Layer 
            l5_filter -= lr * batch_dw5 / batch_sz
            l4_filter -= lr * batch_dw4 / batch_sz
            l3_filter -= lr * batch_dw3 / batch_sz
            l2_filter -= lr * batch_dw2 / batch_sz
            l1_filter -= lr * batch_dw1 / batch_sz
            
            b5 -= lr * batch_db5 / batch_sz
            b4 -= lr * batch_db4 / batch_sz
            b3 -= lr * batch_db3 / batch_sz
            b2 -= lr * batch_db2 / batch_sz
            b1 -= lr * batch_db1 / batch_sz
                # Print average loss per epoch
            print("  \n######Loss per batch###### =>" ,total_loss / batch_sz)
        total_lossst +=total_loss
        print('\n##############loss per epoch#####################',total_lossst/num_batches)
lr = 0.0005
epochs = 100
trainf(lr, epochs )

