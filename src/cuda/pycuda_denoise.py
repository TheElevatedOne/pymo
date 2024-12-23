import cv2
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class PyCudaDenoise:
    """KNN Denoise filter for monochrome image V1.1
    Based on KNN filter CUDA program provided by Nvidia - CUDA SDK samples
    Alain PAILLOU - August 2019
    TheElevatedOne - December 2024 - Modified for this CLI Program"""
    def __init__(self, img: cv2.Mat, model: str) -> None:
        drv.init()
        device = drv.Device(0)
        self.ctx = device.make_context()
        self.img = img
        self.knn = SourceModule("""
__global__ void KNN_Mono(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 2;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 2;
    const float limymin = NLM_BLOCK_RADIUS + 2;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 2;
   
    
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;
        float clr00 = 0.0;
        float clrIJ = 0.0;
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4];

        for(float i = -NLM_BLOCK_RADIUS; i <= NLM_BLOCK_RADIUS; i++)
            for(float j = -NLM_BLOCK_RADIUS; j <= NLM_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ = img_r[index2];
                float distanceIJ = ((clrIJ - clr00) * (clrIJ - clr00)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr += clrIJ * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr = clr + (clr00 / 256.0 - clr) * lerpQ;
        
        dest_r[index5] = (int)(clr * 256.0);
    }
}
""")
        self.nlm2 = SourceModule("""
__global__ void NLM2_Mono(unsigned char *dest_r, unsigned char *img_r,
int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 2;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 2;
    const float limymin = NLM_BLOCK_RADIUS + 2;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 2;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ = img_r[index3];
                
                clr += clrIJ * weightIJ;
 
                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float clr00 = 0.0;
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4] / 256.0;
        
        clr = clr + (clr00 - clr) * lerpQ;
       
        dest_r[index5] = (int)(clr * 256.0);
    }
}
""")
        self.model = model

    def run(self) -> np.ndarray:
        if self.model == "nlm":
            mono_gpu = self.nlm2.get_function("NLM2_Mono")
        else:
            mono_gpu = self.knn.get_function("KNN_Mono")

        wv, wh = self.img.shape

        nb_threads_x = 8
        nb_threads_y = 8
        nb_blocks_x = (wv // nb_threads_x) + 1
        nb_blocks_y = (wh // nb_threads_y) + 1

        gpu_noise = 0.32
        if self.model == "nlm":
            gpu_noise = 1.45
        noise = 1.0 / (gpu_noise * gpu_noise)
        lerpc = 0.2

        r_gpu = drv.mem_alloc(self.img.size * self.img.dtype.itemsize)
        drv.memcpy_htod(r_gpu, self.img)
        img_r_gpu = drv.mem_alloc(self.img.size * self.img.dtype.itemsize)
        drv.memcpy_htod(img_r_gpu, self.img)
        res_r = np.empty_like(self.img)

        mono_gpu(r_gpu, img_r_gpu, np.intc(wv), np.intc(wh), np.float32(noise),
                     np.float32(lerpc), block=(nb_threads_x, nb_threads_y, 1),
                     grid=(nb_blocks_x, nb_blocks_y))

        drv.memcpy_dtoh(res_r, r_gpu)
        r_gpu.free()
        img_r_gpu.free()

        self.ctx.pop()

        return res_r
