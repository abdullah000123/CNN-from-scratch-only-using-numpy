import numpy as np
import os 
import cv2 as cv
import matplotlib.pyplot as plt

print('loading data')
data = 'data/train'  # Path to the dataset
labels = []
images = []
image_size = 240

# Loop through the dataset folder
for imagefile in os.listdir(data):
    image_path = os.path.join(data, imagefile)
    image = cv.imread(image_path, cv.IMREAD_COLOR)  # Read image
    if image is None:
        print(f"Failed to load image: {imagefile}")
        continue  # Skip if image fails to load

    image = cv.resize(image, (image_size, image_size)).astype(np.float16)/255 # Resize image to (120, 120)

    # Assign labels based on file name
    if imagefile.startswith("c"):
        label = 0
    elif imagefile.startswith("d"):
        label = 1
    else:
        continue  # Skip if the file doesn't match the label criteria

    labels.append(label)
    images.append(image)

print('Labels count:', len(labels))
print('Images count:', len(images))

# Convert lists to NumPy arrays
labels = np.array(labels)
images = np.array(images)

# Convert labels to one-hot encoding
num_classes = 2  # Number of classes
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


batch_sz = 5
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
batch_img = np.array(batch_img).astype(np.float32)
batch_lab = np.array(batch_lab)
print(batch_img[0])
print(batch_lab[0])
plt.imshow(batch_img[10 , 0])
print('label' ,batch_lab[10 ,0])
class Activation:
    def relu_derivative(self ,da ,z):
        return da*(z > 0)
    # Activation function
    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self,a):
        return a * (1 - a)
        
    def relu(self,x):
        return np.maximum(0, x)
        
    def relu_derivatived(self ,x):
        return (x > 0)
            
    def softmax(self ,x):
        exp_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return exp_x / (np.sum(exp_x, axis = 1, keepdims = True) + 0.0000005)
        
############################################################################################################

class LayerNormalizationSingleBatch:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def forward(self, x):

        # Compute mean and variance over the (c, h, w) dimensions
        self.mean = np.mean(x, axis=(0 ,1, 2), keepdims=True)  # Shape: (1, 1, 1)
        self.variance = np.var(x, axis=(0 ,1, 2), keepdims=True)  # Shape: (1, 1, 1)
        self.x_normalized = (x - self.mean) / np.sqrt(self.variance + self.epsilon)  # Normalize
        return self.x_normalized

    def backward(self, dout):
     
        c, h, w = dout.shape
        num_elements = c * h * w  # Total number of elements in (c, h, w)

        # Gradient of normalized input
        dx_normalized = dout

        # Gradient of variance
        dvar = np.sum(dx_normalized * (self.x_normalized * -0.5) / np.sqrt(self.variance + self.epsilon), axis=(0 ,1, 2), keepdims=True)

        # Gradient of mean
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.variance + self.epsilon), axis=(0 ,1, 2), keepdims=True) \
                + dvar * np.sum(-2 * (self.x_normalized), axis=(0 ,1, 2), keepdims=True) / num_elements

        # Gradient of input
        dx = dx_normalized / np.sqrt(self.variance + self.epsilon) \
             + dvar * 2 * (self.x_normalized) / num_elements \
             + dmean / num_elements

        return dx
        
class LayerNormalization_dense:
    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def forward(self, x):
    
        self.mean = np.mean(x, axis=-1, keepdims=True)  # Compute mean along the last axis
        self.variance = np.var(x, axis=-1, keepdims=True)  # Compute variance along the last axis
        self.x_normalized = (x - self.mean) / np.sqrt(self.variance + self.epsilon)  # Normalize
        return self.x_normalized

    def backward(self, dout):
        
        N = dout.shape[-1]  # Number of features
        dx_normalized = dout
        dvar = np.sum(dx_normalized * (self.x_normalized * -0.5) / np.sqrt(self.variance + self.epsilon), axis=-1, keepdims=True)
        dmean = np.sum(dx_normalized * -1 / np.sqrt(self.variance + self.epsilon), axis=-1, keepdims=True) + \
                dvar * np.sum(-2 * (self.x_normalized), axis=-1, keepdims=True) / N
        dx = dx_normalized / np.sqrt(self.variance + self.epsilon) + dvar * 2 * (self.x_normalized) / N + dmean / N
        return dx

########################################################################################################

class convolution:        
    def conv2(self ,l1_filter ,images ,b ,layer):
        if layer == 1:
            stride = 4
            image_height, image_width, image_channels = images.shape
        else:   
            stride = 1
            image_channels , image_height , image_width = images.shape
            
        no_of_f ,  f_size_h , f_size_w = l1_filter.shape
        
        # If padding is applied, pad the image
        if padding > 0:
            images = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        
        # Compute the output dimensions
        output_height = (image_height - f_size_h) // stride + 1
        output_width = (image_width - f_size_w) // stride + 1
        
        # Initialize the output feature map
        self.output = np.zeros((l1_filter.shape[0] ,output_height ,output_width ,3))
        self.output_of_layer = np.zeros((no_of_f ,output_height , output_width))
        if layer == 1:
             for k in range(l1_filter.shape[0]):
                for c in range(3):
                    for i in range(0, output_height):
                        for j in range(0, output_width):
                            #region = region = images[i:i+l1_filter[0].shape[0], j:j+l1_filter[0].shape[1],c]
                            self.output[k ,i ,j , c] = np.sum(images[i:i+l1_filter[0].shape[0] ,j:j+l1_filter[0].shape[1] ,c] * l1_filter[k])
                self.output_of_layer = np.sum(self.output ,axis=3)
        else:
            for k in range(no_of_f):
                for h in range(output_height):
                    for w in range(0, output_width):
                        self.output_of_layer[k ,h ,w] = np.sum(images[: ,h:h + f_size_h  ,w:w + f_size_w] * l1_filter[k])
                           
        return self.output_of_layer + b

    def conv_backward(self ,dZ ,weight ,input_maps ,layer ,padding=0):
        A_prev = input_maps
        (pre_map, n_H_prev, n_W_prev) = A_prev.shape
        (no_filters, f, f) = weight.shape #filter of current layer
        (m ,nh ,nw) = dZ.shape       #derivative of loss wrt ouput
        #print('dz and input to this layer in back propogation maps',dZ.shape[0],A_prev.shape[0])
        #print('no of filter',weights.shape[0])
        if layer == 1:
            stride = 4 
            pre_map = 3
            n_H_prev = A_prev[0]
            n_W_prev = A_prev[1]
        else:
            (pre_map, n_H_prev, n_W_prev) = A_prev.shape
            stride = 1 
        self.dA_prev = np.zeros_like(A_prev)
        self.dW = np.zeros_like(weight)
        self.db = np.zeros((no_filters , 1 ,1))
        
        for c in range(no_filters): 
            for h in range(nh):               
                for w in range(nw): 
                    
                    vert_start = h * stride 
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
             
                    if layer!=1:
                            # gradient of sum (f*a_slide) = sum of gradients (f*dZ)
                        a_slice = A_prev[: ,vert_start: vert_end , horiz_start:horiz_end]
                        a_slice = np.sum(a_slice ,axis=0)
                        self.dA_prev[: ,vert_start:vert_end , horiz_start:horiz_end] += (weight[c ,: ,:] * dZ[c ,h ,w])    
                        #self.dA_prev[: ,vert_start:vert_end , horiz_start:horiz_end] = np.sum((weight[c ,: ,:] * dZ[c ,h ,w]))                

                    else:
                        a_slice = A_prev[vert_start: vert_end , horiz_start:horiz_end ,:]
                        a_slice = np.sum(a_slice ,axis=2)
                        #dA_prev[vert_start:vert_end, horiz_start:horiz_end,:] += (weight[c,:,:] * dZ[c,h,w]).astype(np.uint8)
           
                    self.dW[c ,: ,:] += a_slice * dZ[c ,h ,w]
                    #db +=np.sum(dZ[c,h,w])
            self.db[c ,: ,:] =np.sum( dZ[c, :, :])

  
                      #db[c,:,:]+=dZ[c,h,w]
        return self.dA_prev, self.dW, self.db
        
########################################################################################################  

class pooling:
    def __init__(self ,images ,s ,ps):
        self.s = 2
        self.ps = 3
        self.images = images
        self.maps, self.height, self.width = images.shape
        self.pool_height, self.pool_width = self.ps, self.ps
        self.output_height = (self.height - self.pool_height) // self.s + 1
        self.output_width = (self.width - self.pool_width) // self.s + 1
        self.pooling_out = np.zeros((self.maps, self.output_height, self.output_width))
        self.positions_row = np.zeros((self.maps, self.output_height, self.output_width), dtype=int)
        self.positions_col = np.zeros((self.maps, self.output_height, self.output_width), dtype=int)
        self.cache_p = (images, self.positions_row, self.positions_col)
        
    def update_parms(self ,images ,s ,ps):
        self.images = images
        self.ps = ps
        self.s = s
        self.maps, self.height, self.width = self.images.shape
        self.pool_height, self.pool_width = self.ps, self.ps
        self.output_height = (self.height - self.pool_height) // self.s + 1
        self.output_width = (self.width - self.pool_width) // self.s + 1
        self.pooling_out = np.zeros((self.maps, self.output_height, self.output_width))
        self.positions_row = np.zeros((self.maps, self.output_height, self.output_width), dtype=int)
        self.positions_col = np.zeros((self.maps, self.output_height, self.output_width), dtype=int)
        self.cache_p = (images, self.positions_row, self.positions_col)
        
    def pooling_max(self):
        for i in range(self.maps):
            for h in range(self.output_height):
                for w in range(self.output_width):
                    vert_start = h * self.s
                    vert_end = vert_start + self.pool_height
                    horiz_start = w * self.s
                    horiz_end = horiz_start + self.pool_width
                    pool_region = self.images[i, vert_start:vert_end, horiz_start:horiz_end]
                    # Find the max value and its index in the pool region
                    max_val = np.max(pool_region)
                    max_row, max_col = np.unravel_index(pool_region.argmax(), pool_region.shape)
                    # Store the max value and its indices
                    self.pooling_out[i, h, w] = max_val
                    self.positions_row[i, h, w] = vert_start + max_row
                    self.positions_col[i, h, w] = horiz_start + max_col
                    #max_p[i, h, w]=np.unravel_index(pool_region.argmax(), pool_region.shape)
        # Cache for backpropagation
        self.cache_p = (self.images, self.positions_row, self.positions_col)
        #print('max poll shape ',self.pooling_out.shape)
        return self.pooling_out , self.cache_p


    def pooling_backward(self,dA,cache):
        (A_prev, position_row,position_col) = cache
        (m, h_prev, w_prev) = A_prev.shape
        dA_prev = np.zeros_like(A_prev)
        for i in range(m):  
            for h in range(dA.shape[1] ):  
                for w in range(dA.shape[2] ):  
                    # Retrieve the corresponding max positions
                    
                    vert_start = h * self.s
                    horiz_start = w * self.s
                    max_row = position_row[i, h, w]
                    max_col = position_col[i, h, w]
                    dA_prev[i, max_row, max_col] += dA[i, h, w]
        #print('reverse pooling done',dA_prev.shape)
        return dA_prev

########################################################################################################################

class flatten:
    def flatten_forward(self,x):
        # Store the original shape to reshape during backpropagation
        self.original_shape = x.shape
        #print('orignal shape before  forward flatten', original_shape)
        flattened = x.reshape(1, -1) 
        return flattened, self.original_shape
        
    def flatten_backward(self,d_flattened):
        #print('result  ',d_flattened.reshape(original_shape).shape)
        return d_flattened.reshape(self.original_shape)

########################################################################################################################

class dense:
    # Forward propagatio        
    def forward_propagation_dense(self, x):
        #self.a1, self.a2, self.a3 ,self.a4
        z11 = np.dot(x, theta0) + bias0
        z11 = norm_d1.forward(z11)
        self.a1 = relu_1.relu(z11)
        #print('a1',a1.shape)
    
        z22 = np.dot(self.a1, theta1) + bias1
        z22 = norm_d2.forward(z22)
        self.a2 = relu_1.relu(z22)
        #print('a2',a2.shape)
    
        z33 = np.dot(self.a2, theta2) + bias2
        z33 = norm_d3.forward(z33)
        self.a3 = relu_1.relu(z33)
        
        z44 = np.dot(self.a3, theta3) + bias3
        print('input to softmax ',z44)
        self.a4 = relu_1. softmax(z44)
       
        return self.a4
    
        
    def back_propagation_dense(self, x, y):
        ls = self.a4 - y
        #print('a3',a3.shape, a3)
        #print('y',y.shape, y)
        dZ4 = ls  
        #print('dZ3',dZ3)
        dA3 = np.dot(dZ4, theta3.T) 
        dA3 = norm_d3.backward(dA3)
        dZ3 = dA3 * relu_1.relu_derivatived(self.a3)
    
        dA2 = np.dot(dZ3, theta2.T) 
        dA2 = norm_d2.backward(dA2)
        dZ2 = dA2* relu_1.relu_derivatived(self.a2)
        #print('dZ2',dZ2.shape)
    
        dA1 = np.dot(dZ2, theta1.T)
        dA1 = norm_d1.backward(dA1)
        dZ1 = dA1 * relu_1.relu_derivatived(self.a1)
        #print('dA1',dA1.shape)
    
        dA0 = np.dot(dZ1, theta0.T) 
        #print('dA0',dA0.shape)
        
        dtheta3 = np.dot(self.a3.T, dZ4)
        dbias3 = np.sum(dZ4, axis=0, keepdims=True)
        
        dtheta2 = np.dot(self.a2.T, dZ3)
        dbias2 = np.sum(dZ3, axis=0, keepdims=True)
    
        dtheta1 = np.dot(self.a1.T, dZ2)
        dbias1 = np.sum(dZ2, axis=0, keepdims=True)
    
        dtheta0 = np.dot(x.T, dZ1)
        dbias0 = np.sum(dZ1, axis=0, keepdims=True)
        
        return dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2 ,dtheta3 ,dbias3 ,dA0
        


    
##########################################################################################################
def conv_forward(batch_im):
    global cache  ,cache2  ,cache5  ,result ,result_2 ,result_3 ,result_4 ,result_5 ,pooling_5   ,pooling_2 ,pooling_1 ,result_relu4 ,result_relu3
    
    result = layer_1.conv2(l1_filter ,batch_im ,b1 ,1)
    result_relu = relu_1.relu(result)
    r1 = norm1.forward(result_relu)
    
    pool.update_parms(r1 , s , ps)
    pooling_1 , cache = pool.pooling_max()
    #print('max 1 out',pooling_1)
    
    result_2 = layer_1.conv2(l2_filter ,pooling_1 ,b2 ,2)
    result_relu2 = relu_1.relu(result_2)
    r2 = norm2.forward(result_relu2)

    pool.update_parms(r2 , s , ps)
    pooling_2 , cache2 = pool.pooling_max()
    #print('max 2 out',pooling_2)

    result_3 = layer_1.conv2(l3_filter ,pooling_2 ,b3 ,3)
    result_relu3 = relu_1.relu(result_3)
    r3 = norm3.forward(result_relu3)

    
    result_4 = layer_1.conv2(l4_filter ,r3 ,b4 ,4)
    result_relu4 = relu_1.relu(result_4)
    r4 = norm4.forward(result_relu4)
    #print('conv 3 4 out',r4)

    result_5 = layer_1.conv2(l5_filter ,r4 ,b5 ,5)
    result_relu5 = relu_1.relu(result_5)
    r5 = norm5.forward(result_relu5)

    pool.update_parms(r5 , s , ps)
    pooling_5 , cache5  = pool.pooling_max()
    
    conv_result , orignal_shape = flat.flatten_forward(pooling_5)
    #print('conv output',conv_result)
    return conv_result , orignal_shape

    

def cnn_backpropogation(dA0, image):
    d1 = flat.flatten_backward(dA0)
    
    #print('reshape     ' , d1.shape)
    d2 = pool.pooling_backward(d1 , cache5 )
    
    r_back5 = norm5.backward(d2)
    d3 = relu_1.relu_derivative(r_back5 , result_5)
    l5_c ,dw5 ,db5 = layer_1.conv_backward(d3 , l5_filter , result_relu4 , 5) # gradient wrt out ,filter used ,input to layer ,layer no

    r_back4 = norm4.backward(l5_c)
    d4 = relu_1.relu_derivative(r_back4 , result_4)   #gradient wrt out ,input to relu          
    l4_c ,dw4 ,db4 = layer_1.conv_backward(d4 , l4_filter , result_relu3 , 4)  # gradient wrt out ,filter used ,input to layer ,layer no

    r_back3 = norm3.backward(l4_c)
    d5 = relu_1.relu_derivative(r_back3 , result_3)    #gradient wrt out ,input to relu      
    l3_c ,dw3 ,db3 = layer_1.conv_backward(d5 , l3_filter , pooling_2 , 3) # gradient wrt out ,filter used ,input to layer ,layer no

    d6 = pool.pooling_backward(l3_c , cache2 )
    r_back2 = norm2.backward(d6)
    d7 = relu_1.relu_derivative(r_back2 ,result_2)           #gradient wrt out ,input to relu      
    l2_c ,dw2 ,db2 = layer_1.conv_backward(d7 , l2_filter , pooling_1 , 2) # gradient wrt out ,filter used ,input to layer ,layer no

    d8 = pool.pooling_backward(l2_c , cache )
    r_back1 = norm1.backward(d8)
    d9 = relu_1.relu_derivative(r_back1 ,result)          #gradient wrt out ,input to relu      
    l1_c ,dw1 ,db1 = layer_1.conv_backward(d9 , l1_filter , image , 1) # gradient wrt out ,filter used ,input to layer ,layer no
    #print('input/back conv 1 shape',l1_c.shape)

    return dw5 ,dw4 ,dw3 ,dw2 ,dw1 ,db5 ,db4 ,db3 ,db2 ,db1


def categorical_cross_entropy_loss(y_pred, y_true):

    # Add a small epsilon to avoid log(0)
    epsilon = 1e-9
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # Compute the categorical cross-entropy loss
    loss = -np.sum(y_true * np.log(y_pred), axis=1)
    # Return the loss
    return loss


#################################################################################################################

def trainf( lr, epochs):
    global theta1, theta2, bias1, bias2, theta0, bias0, theta3, bias3, l1_filter, l2_filter, l3_filter ,l5_filter ,l4_filter , b1, b2, b3 ,b4 ,b5 ,ps ,s
    global norm1 ,norm2 ,norm3 ,norm4 ,norm5 ,norm_d1 ,norm_d2 ,norm_d3 ,norm_d4
    global layer_1 , relu_1 , flat , pool , den 

    total_loss = 0
    batch_imgsss = batch_img[0]
    batch_lbss = batch_lab[0]
    s = 2
    ps = 3
    print('starting first ')
    
    layer_1 = convolution()
    relu_1 = Activation()
    flat = flatten()
    pool = pooling(batch_imgsss[0] ,s ,ps)
    den = dense()

    norm1 = LayerNormalizationSingleBatch()
    norm2 = LayerNormalizationSingleBatch()
    norm3 = LayerNormalizationSingleBatch()
    norm4 = LayerNormalizationSingleBatch()
    norm5 = LayerNormalizationSingleBatch()

    norm_d1 = LayerNormalization_dense()
    norm_d2 = LayerNormalization_dense()
    norm_d3 = LayerNormalization_dense()
    norm_d4 = LayerNormalization_dense()
    
    no_l2 = 128       # layer 1
    no_l3 = 64
    no_l4 = 32         
    no_out = 2      # Output layer for 2 classes
    #######################################
    l1_filter = ( np.random.randn(96 ,11 ,11))
    b1 = np.zeros((96 ,1 , 1))
    
    l2_filter = ( np.random.randn(256 ,5 ,5))
    b2 = np.zeros((256 ,1 , 1))
    
    l3_filter = ( np.random.randn(384 ,3 ,3))
    b3 = np.zeros((384 ,1 , 1))

    l4_filter = ( np.random.randn(384 ,3 ,3))
    b4 = np.zeros((384 ,1 , 1))
    
    l5_filter = ( np.random.randn(256 ,3 ,3))
    b5 = np.zeros((256 ,1 , 1))

    
    #l4_filter=np.random.randn(32,3,3)*0.01
    #b4=np.zeros((32,1, 1))
    ######################################
    bias0 = np.zeros((1, no_l2))
    theta1 = ( np.random.rand(no_l2, no_l3))
    bias1 = np.zeros((1, no_l3))
    theta2 = ( np.random.rand(no_l3, no_l4) )
    bias2 = np.zeros((1, no_l4))
    theta3 = ( np.random.rand(no_l4, no_out) )
    bias3 = np.zeros((1, no_out))  
    ########################################
    result ,orignal_shape = conv_forward(batch_imgsss[0])
    no_input = result.shape[1]   
    
    inputs = result.reshape(-1, no_input)

    theta0 =np.abs( np.random.rand(no_input, no_l2) )
    outputs = den.forward_propagation_dense( inputs)
    
    loss = categorical_cross_entropy_loss( outputs, batch_lbss[0])
    
    
    dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2, dtheta3, dbias3, dA0 = den.back_propagation_dense( inputs , batch_lbss[0] )
    print('dao',dA0.shape)
    dw5 ,dw4, dw3 ,dw2 ,dw1 ,db5 ,db4 ,db3 ,db2 ,db1 = cnn_backpropogation(dA0, batch_imgsss[0])   #gradient and imput image to cn
    print('#######loss###### =>',loss)

    #################################################################################
    for i in range(epochs):
        print('\n############################## epoch no ',i+1)
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
            
            batch_dw5=np.zeros_like(dw5)
            batch_dw4=np.zeros_like(dw4)
            batch_dw3 = np.zeros_like(dw3)
            batch_dw2 = np.zeros_like(dw2)
            batch_dw1 = np.zeros_like(dw1)
            
            batch_db5 = np.zeros_like(db5)
            batch_db4 = np.zeros_like(db4)
            batch_db3 = np.zeros_like(db3)
            batch_db2 = np.zeros_like(db2)
            batch_db1 = np.zeros_like(db1)
            print('\nStarting batch => ',ba + 1)
            for im in range(batch_sz):
        ######                                              CONV FORWARD                                       ##############
                result ,orignal = conv_forward(batch_im[im])
                inputs = result.reshape(-1, no_input)
                outputs = den.forward_propagation_dense( inputs)
                print('Actual is => ', batch_ll[im] )
                print('Predicted is => ',outputs)
                
                loss = categorical_cross_entropy_loss( outputs, batch_ll[im] )
                total_loss += loss  
                
                dtheta0, dbias0, dtheta1, dbias1, dtheta2, dbias2, dtheta3, dbias3, da0 = den.back_propagation_dense( inputs , batch_ll[im] )
            ##################################################################################
                 # Summing up gradients for average
                batch_dtheta0 += dtheta0
                batch_dbias0 += dbias0
                batch_dtheta1 += dtheta1
                batch_dbias1 += dbias1
                batch_dtheta2 += dtheta2
                batch_dbias2 += dbias2
                batch_dtheta3 += dtheta3
                batch_dbias3 += dbias3
                #print(f"gradient wrt last layer theta {dtheta0}")
                dw5 ,dw4, dw3 ,dw2 ,dw1 ,db5 ,db4 ,db3 ,db2 ,db1 = cnn_backpropogation(da0 , batch_im[im])

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
                print('#######loss###### =>',loss)

                 # Updating Dense Layer weights and biases
            theta0 -= lr * batch_dtheta0 / batch_sz
            bias0 -= lr * batch_dbias0 / batch_sz
            theta1 -= lr * batch_dtheta1 / batch_sz
            bias1 -= lr * batch_dbias1 / batch_sz
            theta2 -= lr * batch_dtheta2 / batch_sz
            bias2 -= lr * batch_dbias2 / batch_sz
            theta3 -= lr * batch_dtheta3 / batch_sz
            bias3 -= lr * batch_dbias3 / batch_sz
            
            
                # Updating filters for Convolutional Layer 

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
                # Printing average loss per epoch
            print(" \n ######################Loss per batch############### =>" ,total_loss/batch_sz)
            '''
            print('batch_dtheta3',batch_dtheta3)
            print('batch_dtheta2',batch_dtheta2)
            print('batch_dtheta1',batch_dtheta1)
            print('batch_dtheta0',batch_dtheta0)

            print('batch batch_dw5',batch_dw5)
            print('batch batch_dw3',batch_dw3)
            print('batch batch_dw1',batch_dw1)

            print('theta 0',theta0)
            print('theta 3',theta3)'''

            #print('batch_dtheta0',batch_dtheta0)

lr = 0.0001
epochs = 10
trainf(lr, epochs )
