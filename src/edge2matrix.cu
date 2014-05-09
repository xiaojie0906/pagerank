#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>

typedef struct
{
    int row_i;
    int col_i;
    float value_f;
}spnode;
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }
}
int cmp(const void *a,const void*b)
{

    if( (*(spnode *) a).row_i<(* (spnode *)b).row_i)
        return -1;
    else if( (*(spnode *) a).row_i>(* (spnode *)b).row_i)
        return 1;
    else if( (*(spnode *) a).row_i==(* (spnode *)b).row_i)
        return (*(spnode*)a).col_i- (*(spnode*)b).col_i;

    return 0;
}
//__global__ void sumdiff(float * array,int length)
//{
//    __shared__  tmp[];//2*ThreadPerBlock
//    glid=threadIdx.x+blockDim.x*blockIdx.x;
//    blid=blockIdx.x;
//    thid=threadIdx.x;
//    tmp[thid]=array[glid];
//    
//
//    
//    
//    
//    }
__global__ void sumpr(int * row,int * col,float * outlink,float* pagerank,float *pagerank_new,int nodes,int edges,int totalThread)
{



    int thid=blockDim.x*blockIdx.x+threadIdx.x;
    int edgesPerThread=edges/totalThread;
    int other=edges%totalThread;
    if(thid==totalThread-1)
        edgesPerThread+=other;
    int i;
if(nodes>totalThread)
{
    int nodesPerThread=nodes/totalThread;
    int othernode=nodes%totalThread;
    if(thid==totalThread-1)
        nodesPerThread+=othernode;
    if(thid<totalThread-1)
        for(i=0;i<nodesPerThread;i++)
            pagerank_new[i+thid*nodesPerThread]=0;
    else
        for(i=0;i<nodesPerThread;i++);
          //  pagerank_new[i+thid*(nodesPerThread-other)]=0;
}
else
    if(thid<nodes)
pagerank_new[thid]=0;

    if(thid!=totalThread-1)
        for(i=0;i<edgesPerThread;i++)//every thread compute nodePerThread 
            atomicAdd(pagerank_new+(row[thid*edgesPerThread+i]),outlink[i+thid*edgesPerThread]*pagerank[col[i+edgesPerThread*thid]]);
    // atomicAdd(pagerank_new+(row[thid*edgesPerThread+i]),0.1);
    else
        for(i=0;i<edgesPerThread;i++)//every thread compute nodePerThread 
            atomicAdd(pagerank_new+(row[thid*(edgesPerThread-other)+i]),outlink[i+thid*(edgesPerThread-other)]*pagerank[col[i+(edgesPerThread-other)*thid]]);

}
__global__ void sumpr2(float * pagerank,float * pagerank_new,float ebn,int nodes,int totalThread,float*diffarray)
{
    int thid=blockDim.x*blockIdx.x+threadIdx.x;
    if(nodes>totalThread)
    {
        int nodesPerThread=nodes/totalThread;
        int other=nodes%totalThread;
        if(thid==totalThread-1)
            nodesPerThread+=other;
        int i;
        //计算pr=0.9*sum+0.1ebn
        if(thid<totalThread-1)
            for(i=0;i<nodesPerThread;i++)//every thread compute nodePerThread 
            {
                pagerank_new[i+thid*nodesPerThread]= pagerank_new[i+thid*nodesPerThread]*0.9+0.1*ebn;
                if(pagerank_new[i+thid*(nodesPerThread-other)]>pagerank[i+thid*(nodesPerThread-other)])
                    diffarray[thid]+=pagerank_new[i+thid*nodesPerThread]- pagerank[i+thid*nodesPerThread];
                else
                    diffarray[thid]+=pagerank[i+thid*nodesPerThread]- pagerank_new[i+thid*nodesPerThread];

            }
        else
            for(i=0;i<nodesPerThread;i++)//every thread compute nodePerThread 
            {
                pagerank_new[i+thid*(nodesPerThread-other)]= pagerank_new[i+thid*(nodesPerThread-other)]*0.9+0.1*ebn;
                if(pagerank_new[i+thid*(nodesPerThread-other)]>pagerank[i+thid*(nodesPerThread-other)])
                    diffarray[thid]+=pagerank_new[i+thid*(nodesPerThread-other)]- pagerank[i+thid*(nodesPerThread-other)];
                else
                    diffarray[thid]+=pagerank[i+thid*nodesPerThread]- pagerank_new[i+thid*nodesPerThread];
            }
    }
    else

        if(thid<nodes)

        {        
            pagerank_new[thid]= pagerank_new[thid]*0.9+0.1*ebn;
            if(pagerank_new[thid]>pagerank[thid])
                diffarray[thid]+=pagerank_new[thid]- pagerank[thid];
            else
                diffarray[thid]+=pagerank[thid]- pagerank_new[thid];

        }        


    /*
       __syncthreads();
    //计算两次迭代差值
    //float diff;
    if(thid!=totalThread-1)
    for(i=0;i<nodesPerThread;i++)//every thread compute nodePerThread 
    diffarray[thid]+=pagerank_new[i+thid*nodesPerThread]- pagerank[i+thid*nodesPerThread];
    else
    for(i=0;i<nodesPerThread;i++)//every thread compute nodePerThread 
    diffarray[thid]+=pagerank_new[i+thid*(nodesPerThread-other)]- pagerank[i+thid*(nodesPerThread-other)];
     */
}

__global__ void step3(float * pagerank,float * pagerank_new,float ebn,int nodes,int totalThread,float*diffarray)
{
    int thid=blockDim.x*blockIdx.x+threadIdx.x;
    //计算差值之和
    //    int stride=0;
    //    if(nodes>=totalThread)
    //        for(stride=totalThread/2;stride>0;stride/=2){
    //            __syncthreads();
    //            if(thid<stride)
    //                diffarray[thid]+=diffarray[thid+stride];
    //        }
    //    else
    //        for(stride=nodes/2;stride>0;stride/=2){
    //            __syncthreads();
    //            if(thid<stride)
    //                diffarray[thid]+=diffarray[thid+stride];
    //        }
    //

    //更新pagerank
    if(nodes>totalThread)
    {
        int nodesPerThread=nodes/totalThread;
        int other=nodes%totalThread;
        if(thid==totalThread-1)
            nodesPerThread+=other;
        int i;
        if(thid!=totalThread-1)
            for(i=0;i<nodesPerThread;i++)//every thread compute nodePerThread 
                pagerank[i+thid*nodesPerThread]= pagerank_new[i+thid*nodesPerThread];
        else
            for(i=0;i<nodesPerThread;i++)//every thread compute nodePerThread 
                pagerank[i+thid*(nodesPerThread-other)]= pagerank_new[i+thid*(nodesPerThread-other)];
    }

    else
        if(thid<nodes)
            pagerank[thid]= pagerank_new[thid];
}

int main(int argc,char * argv[])
{
    char filename[]="./document/web-Google.txt";
    FILE *fp;
    if((fp=fopen(filename,"r"))==NULL)
    {
        fprintf(stderr,"open file error");
        exit(1);
    }

    int edges=0;
    int nodes=0;
    spnode *spmatrix;
    char ch;
    char str[1024];
    ch=getc(fp);
    while(ch=='#')
    {
        fgets(str,1024-1,fp);
        puts(str);
        ch=getc(fp);
    }
    ungetc(ch,fp);
    fscanf(fp,"%d%d",&nodes,&edges);
    printf("%d %d\n",nodes,edges);
    nodes+=1;
    spmatrix = (spnode*)malloc((edges)*sizeof(spnode));
    printf("%ld",sizeof spmatrix);

    int row,col;
    int i;
    int pre_node,out_link_count=0;
    /**/
    for(i = 0;i<edges;i++){
        if(fscanf(fp,"%d%d",&row,&col)!=2)
            fprintf(stderr,"error while reading edges");
        spmatrix[i].row_i=row;
        spmatrix[i].col_i=col;
        if(pre_node==row)
            out_link_count++;
        else{
            for(int j=1;j<=out_link_count;j++)
            {
                spmatrix[i-j].value_f=(float)1/out_link_count;

            }
            pre_node=row;
            out_link_count=1;
        }
    }
    /*end of file ,add the last value of matrix*/
    for(int j=1;j<=out_link_count;j++)
    {
        spmatrix[edges-j].value_f=(float)1/out_link_count;
    }

    /*交换行列，原来是[i][j]表示从i节点到j节点的链路，交换后形成从节点j到节点i的链路*/
    int temp;
    for(i=0;i<edges;i++)
    {
        temp=spmatrix[i].row_i;
        spmatrix[i].row_i=spmatrix[i].col_i;
        spmatrix[i].col_i=temp;
    }

    qsort(spmatrix,edges,sizeof(spnode),cmp);

    float *pr,*pr_new;
    pr=(float*)malloc(sizeof(double)*nodes);
    pr_new=(float*)malloc(sizeof(double)*nodes);
    if(pr==NULL||pr_new==NULL)
        fprintf(stderr,"malloc error");
    for(i=0;i<nodes;i++)
    {
        pr[i]=(float)10;
        pr_new[i]=0;
    }

    float ebn=(double)1/nodes;
    printf("ebn=%.15f\n",ebn);
    int index=0;
    float sum=0.0;
    float convergence=10;
    int iter_times=0;
    struct timeval starttime,endtime;
    gettimeofday(&starttime,NULL);
    puts("start timing\n");
    while(convergence>1000)
    {
        printf("time %d\n",iter_times);
        iter_times++;
        // nimei!!!!!!!!!!!!!!!
        index=0;
        ///////////////////////

        for(i=0;i<nodes;i++)
        {
            sum=0;
            if(spmatrix[index].row_i==i)
            {

                while(spmatrix[index].row_i==i)
                {
                    sum+=spmatrix[index].value_f*pr[spmatrix[index].col_i];
                    index++;
                }
            }
            pr_new[i]=0.9*sum+0.1*ebn;
        }

        // step4
        convergence=0;
        for(int k=0;k<nodes;k++)
            convergence+=fabs(pr_new[k]-pr[k]);

        //change new
        for(int k=0;k<nodes;k++)
            pr[k]=pr_new[k];

        printf("convergence = %.15f\n",convergence);

    }



    gettimeofday(&endtime,NULL);
    long elapsetime = 1000000*(endtime.tv_sec-starttime.tv_sec)+endtime.tv_usec-starttime.tv_usec;
    long mstime=elapsetime/1000;
    printf("time : %lds%ldms\n",mstime/1000,mstime%1000);

    FILE *prFile;
    if((prFile=fopen("pr.txt","w"))==NULL)
        fprintf(stderr,"open sparseMatrix.txt error");
    else
        for(i=0;i<nodes;i++)
            fprintf(prFile,"%.25f\n",pr_new[i]);

    FILE *spMatrixFile;
    if( (spMatrixFile=fopen("sparseMatrix.txt","w"))==NULL)
        fprintf(stderr,"open sparseMatrix.txt error");
    else
        for(i=0;i<edges;i++)
            fprintf(spMatrixFile,"%d  %d  %f\n",spmatrix[i].row_i,spmatrix[i].col_i,spmatrix[i].value_f);
    fclose(fp);
    fclose(spMatrixFile);
    fclose(prFile);


    int * row_d;
    int * col_d;
    float * outlink_d;
    float * pagerank_d;
    float *pagerank_new_d;

    float* diffarray;
    cudaMalloc((void **)&diffarray,sizeof(float)*nodes);

    cudaMalloc((void **)&row_d,sizeof(int)*edges);
    cudaMalloc((void **)&col_d,sizeof(int)*edges);
    cudaMalloc((void **)&outlink_d,sizeof(float)*edges);
    cudaMalloc((void **)&pagerank_d,sizeof(float)*nodes);
    cudaMalloc((void **)&pagerank_new_d,sizeof(float)*nodes);

    int *temparray;
    temparray=(int *)malloc(sizeof(int)*edges);
    float *ftemparray;
    ftemparray=(float*)malloc(sizeof(double)*edges);
    float * pagerank_h;
    pagerank_h=(float*)calloc(nodes,sizeof(double));

    for(i=0;i<edges;i++)
        temparray[i]=spmatrix[i].row_i;
    cudaMemcpy(row_d,temparray,sizeof(int)*edges,cudaMemcpyHostToDevice);

    for(i=0;i<edges;i++)
        temparray[i]=spmatrix[i].col_i;
    cudaMemcpy(col_d,temparray,sizeof(int)*edges,cudaMemcpyHostToDevice);



    for(i=0;i<edges;i++)
        ftemparray[i]=spmatrix[i].value_f;
    cudaMemcpy(outlink_d,ftemparray,sizeof(float)*edges,cudaMemcpyHostToDevice);

    for(i=0;i<nodes;i++)
        ftemparray[i]=(float)10;
    cudaMemcpy(pagerank_d,ftemparray,sizeof(float)*nodes,cudaMemcpyHostToDevice);
    for(i=0;i<nodes;i++)
        ftemparray[i]=0.0;
    cudaMemcpy(pagerank_new_d,ftemparray,sizeof(float)*nodes,cudaMemcpyHostToDevice);
    cudaMemcpy(diffarray,ftemparray,sizeof(float)*nodes,cudaMemcpyHostToDevice);



    int BlockPerGrid = 1024;
    int ThreadPerBlock = 1024;

    dim3 dimGrid (BlockPerGrid);
    dim3 dimBlock ( ThreadPerBlock);
    int iter=161;
    int totalThread=ThreadPerBlock*BlockPerGrid;
    while(iter--){
        printf("nodes:%d totalThread: %d\n",nodes,totalThread);
        checkCUDAError("1");

        sumpr<<< dimGrid,dimBlock>>>(row_d,col_d,outlink_d,pagerank_d,pagerank_new_d,nodes,edges,totalThread);
        cudaThreadSynchronize();
        checkCUDAError("2");
        sumpr2 <<<dimGrid,dimBlock>>>(pagerank_d,pagerank_new_d,(float)ebn,nodes,totalThread,diffarray);
        cudaThreadSynchronize();
        checkCUDAError("3");

        step3<<<dimGrid,dimBlock>>>(pagerank_d,pagerank_new_d,(float)ebn,nodes,totalThread,diffarray);
        cudaThreadSynchronize();
        checkCUDAError("4");
    }
    cudaMemcpy(pagerank_h,pagerank_new_d,nodes*sizeof(float),cudaMemcpyDeviceToHost);
    FILE *gpufp;
    if((gpufp=fopen("gpupr.txt","w"))==NULL)
        fprintf(stderr,"open gpufile error");
    else
        for(i=0;i<nodes;i++)
            fprintf(gpufp,"%.15f\n",pagerank_h[i]);
    fclose(gpufp);

    cudaMemcpy(pagerank_h,pagerank_d,nodes*sizeof(float),cudaMemcpyDeviceToHost);
    //    float tmpf=0.0;
    //    for(i=0;i<nodes;i++)
    //        tmpf+=pagerank_h[i];
    //    printf("sum = %f\n",tmpf);

    FILE *tmpfp;
    if((tmpfp=fopen("tmpfppr.txt","w"))==NULL)
        fprintf(stderr,"open gpufile error");
    else
        for(i=0;i<nodes;i++)
            fprintf(tmpfp,"%.10f\n",pagerank_h[i]);
    fclose(tmpfp);



    cudaFree(row_d);
    cudaFree(col_d);
    cudaFree(outlink_d);
    cudaFree(pagerank_d);
    cudaFree(pagerank_new_d);
    cudaFree(diffarray);
    free(pr);
    free(pr_new);
    free(temparray);
    free(ftemparray);
    free(pagerank_h);
    free(spmatrix);
    return 0;
}




void sort(spnode* spmatrix,int length)
{



}
