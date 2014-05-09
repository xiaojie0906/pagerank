#include<stdio.h>
#include<sys/time.h>
#include<stdlib.h>
#include<math.h>

typedef struct
{
    int row_i;
    int col_i;
    float value_f;
}spnode;

int cmp(const void *a,const void*b)
{

    if( (*(spnode *) a).row_i<(* (spnode *)b).row_i)
        return -1;
    else if( (*(spnode *) a).row_i>(* (spnode *)b).row_i)
        return 1;
    else if( (*(spnode *) a).row_i==(* (spnode *)b).row_i)
        return (*(spnode*)a).col_i- (*(spnode*)b).col_i;
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
    char str[100];
    ch=getc(fp);
    while(ch=='#')
    {
        fgets(str,100-1,fp);
        puts(str);
        ch=getc(fp);
    }
    ungetc(ch,fp);
    fscanf(fp,"%d%d",&nodes,&edges);
    printf("%d %d\n",nodes,edges);
    nodes++;

    spmatrix = (spnode*)malloc((edges)*sizeof(spnode));
    spmatrix = (spnode*)malloc((edges)*sizeof(spnode));
    printf("%ld",sizeof spmatrix);
    int row,col;
    int i,j;
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

   int *r_csr; 
   r_csr=(int *)malloc(sizeof(int)*(nodes+1));
   for(i=0;i<nodes;i++)
       r_csr[i]=-1;
int current_row=0;
int index=0;
   for(i=0;i<edges;)
   {
    if( spmatrix[i].row_i==current_row)
    {
        r_csr[current_row]=i;
        while(spmatrix[i].row_i==current_row)
        i++;
   }
    current_row++;

   }
r_csr[nodes]=edges-1;


    float *pr,*pr_new;
    pr=(float*)malloc(sizeof(float)*nodes);
    pr_new=(float*)malloc(sizeof(float)*nodes);
    if(pr==NULL||pr_new==NULL)
        fprintf(stderr,"malloc error");
    for(i=0;i<nodes;i++)
    {
        pr[i]=(float)10;
        pr_new[i]=0;
    }

    


    float ebn=(float)1/nodes;
    printf("ebn=%.15f\n",ebn);
    float sum=0.0;
    float d;
    float convergence=10;
    int iter_times=0;
    struct timeval starttime,endtime;
    gettimeofday(&starttime,NULL);
    puts("start timing\n");
  //  while(convergence>0.01)
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
                if(i==0)
                    printf("sum = %f , value = %f,pr= %f\n",sum,spmatrix[index].value_f,pr[spmatrix[index].col_i]);
                    index++;
                }
            }
            pr_new[i]=0.9*sum+0.1*ebn;
        }

        /*
        // step1
        for(i=0;i<nodes;i++)
        {
        if(spmatrix[index].row_i==i)
        {

        sum=0;
        while(spmatrix[index].row_i==i)
        {
        sum+=spmatrix[index].value_f*pr[spmatrix[index].col_i];
        index++;
        }
        pr_new[i]=sum;
        }
        else
        pr_new[i]=0;

        }

        //step2
        d=0; 
        for(int k=0;k<nodes;k++)
        d+=fabs(pr[k])-fabs(pr_new[k]);
        printf("d=%f\n",d);


        //step3
        for(i=0;i<nodes;i++)
        pr_new[i]+=d*ebn;
        */
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

    FILE * csrfp;
    if((csrfp=fopen("csr.txt","w"))==NULL)
        fprintf(stderr,"open csr.txt error");
    else
        for(i=0;i<nodes;i++)
            fprintf(csrfp,"%d\n",r_csr[i]);
    fclose(csrfp);




    free(spmatrix);
    fclose(fp);
    fclose(spMatrixFile);
    fclose(prFile);
    return 0;
}

void sort(spnode* spmatrix,int length)
{



}
