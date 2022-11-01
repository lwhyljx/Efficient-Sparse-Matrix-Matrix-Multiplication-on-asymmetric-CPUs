#define __USE_GNU
#include "Originalmat.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>

#define EIGEN_DONT_PARALLELIZE
//#define _GNU_SOURCE

#include<iostream>
#include<fstream>
#include<vector>
#include<cmath>
#include<tuple>
#include<set>
#include<algorithm>
#include<cstring>
#include<random>
#include<time.h>
#include<unistd.h>
#include<sched.h>
#include<ctype.h>
#include<cstring>
#include<pthread.h>
#include<sys/syscall.h>

#define eps 1e-4
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>MatrixXf;
typedef Eigen::Triplet<float> T;
typedef Eigen::SparseMatrix<float,Eigen::RowMajor> SparseMatrixType;


std::tuple<Originalmat,Originalmat,int> ReadData(std::string filename){
  float* data;//存储读入的float数据
  std::vector<int>feature;//存储稀疏稠密矩阵的行列
  std::ifstream infile(filename,std::ios::in);
  std::string temp;//存储文件第一行
  infile>>temp;
  int cnt=0;//从第一行文本获取行列
  int l=temp.length();
  for(int i=0;i<l;i++){
    if(temp[i]==','){
      feature.emplace_back(cnt);
      cnt=0;
    }
    else{
      cnt=cnt*10+temp[i]-'0';
    }
  }
  feature.emplace_back(cnt);
  int H=feature[0];
  int W=feature[1];
  int N=feature[2];
  //std::cout<<temp<<std::endl;
  size_t nSize=static_cast<size_t>(H)*static_cast<size_t>(W);
  data=new float[nSize];
  float x;
  cnt=0;
  int nnz=0;
  while(infile>>x){
    data[cnt++]=x;
    if(std::fabs(x-0)<eps){

    }
    else{
      nnz++;
    }
  }
  Originalmat Sparsemat,Densemat(W,N);
  Sparsemat.setmat(H,W,data);
  //std::cout<<nnz<<std::endl;
  delete []data;
  data=nullptr;
  return std::make_tuple(Sparsemat,Densemat,nnz);
}

std::vector<std::vector<int>> ColIndex(float* M,int row,int col)
{
  std::vector<std::vector<int>> RowColSet;
  RowColSet.resize(row);
  for(int i=0;i<row;i++){
    for(int j=0;j<col;j++){
      if(std::fabs(M[i*col+j]-0.0)>eps){
        RowColSet[i].emplace_back(j);
      }
    }
  }
  return RowColSet;
}

int RowIntersection(std::set<int>& RowUse,int lastRow,std::vector<std::vector<int>>RowColSet)
{
  int Colintersectionnum=-1;
  int row;
  for(auto &it:RowUse){
    std::vector<int>v_intersection;
    std::set_intersection(RowColSet[it].begin(),RowColSet[it].end(),RowColSet[lastRow].begin(),RowColSet[lastRow].end(),std::back_inserter(v_intersection));
    //std::cout<<v_intersection.size()<<std::endl;
    if((int)v_intersection.size()>Colintersectionnum){
      Colintersectionnum=v_intersection.size();
      //std::cout<<row<<std::endl;
      row=it;
    }
  }
  return row;
}

int* WeightReorder(Originalmat& Sparsemat)
{
  float *A,*B;
  int rows,cols;
  int *MappedRow;
  std::set<int>RowUse;
  A=Sparsemat.get_mat();
  rows=Sparsemat.get_row();
  cols=Sparsemat.get_col();
  size_t nSize=static_cast<size_t>(rows)*static_cast<size_t>(cols);
  size_t nRow=static_cast<size_t>(rows);
  B=new float[nSize];
  MappedRow=new int[nRow];
  for(int i=0;i<rows;i++){
    RowUse.insert(i);
  }
  std::vector<std::vector<int>> RowColSet=ColIndex(A,rows,cols);
  int row=0;
  int nnznum=RowColSet[0].size();
  for(int i=1;i<rows;i++){
    int colnum=RowColSet[i].size();
    if(nnznum<colnum){
      nnznum=colnum;
      row=i;
    }
  }
  memcpy(B,A+row*cols,sizeof(float)*cols);
  MappedRow[0]=row;
  RowUse.erase(row);
  for(int i=1;i<rows;i++){
    //std::cout<<1<<std::endl;
    row=RowIntersection(RowUse,row,RowColSet);
    memcpy(B+i*cols,A+row*cols,sizeof(float)*cols);
    MappedRow[i]=row;
    RowUse.erase(row);
  }
  Sparsemat.setmat(rows,cols,B);
  //std::cout<<B<<" "<<Sparsemat.get_mat()<<std::endl;
  ///delete [] A;
  delete []B;
  B=nullptr;
  return MappedRow;
}

void init_Densematrix(Originalmat& Densemat)
{
  std::default_random_engine engin;
  engin.seed(time(nullptr)); 
  std::uniform_real_distribution<float> dist(0,1);
  float* M;
  int row=Densemat.get_row();
  int col=Densemat.get_col();
  size_t nSize=static_cast<size_t>(row)*static_cast<size_t>(col);
  M=new float[nSize];
  for(int i=0;i<row*col;i++)
  {
    M[i]=dist(engin);
  }
  Densemat.setmat(row,col,M);
  delete []M;
}

std::pair<Originalmat,Originalmat> DenseClusterPartition(Originalmat& Densemat,float radio)
{
  int row,col;
  float* M;
  row=Densemat.get_row();
  col=Densemat.get_col();
  int colSplit=(int) round(radio*col);
  int Bigcol=colSplit,Littlecol=col-colSplit;
  Originalmat BigDensemat(row,Bigcol),LittleDensemat(row,Littlecol);
  return std::make_pair(BigDensemat,LittleDensemat);
}

std::tuple<SparseMatrixType,MatrixXf,MatrixXf> ConvertFormat(Originalmat& Sparsemat,Originalmat& BigDensemat,Originalmat& LittleDensemat,int nnz)
{
  int Sparsematrow=Sparsemat.get_row(),Sparsematcol=Sparsemat.get_col();
  int BigDensematrow=BigDensemat.get_row(),BigDensematcol=BigDensemat.get_col();
  int LittleDensematrow=LittleDensemat.get_row(),LittleDensematcol=LittleDensemat.get_col();
  float *M=Sparsemat.get_mat();
  std::srand((unsigned int) time(0));
  MatrixXf BigDensematrix=Eigen::MatrixXf::Random(BigDensematrow,BigDensematcol);
  MatrixXf LittleDensematrix=Eigen::MatrixXf::Random(LittleDensematrow,LittleDensematcol);
  SparseMatrixType SparseMatrix(Sparsematrow,Sparsematcol);
  std::vector<T> tripletList;
  tripletList.reserve(nnz);
  M=Sparsemat.get_mat();
  for(int i=0;i<Sparsematrow;i++){
    for(int j=0;j<Sparsematcol;j++){
      if(fabs(M[i*Sparsematcol+j]-0)>eps){
        tripletList.push_back(T(i,j,M[i*Sparsematcol+j]));
      }
    }
  }
  SparseMatrix.setFromTriplets(tripletList.begin(), tripletList.end());
  SparseMatrix.makeCompressed();
  return std::make_tuple(SparseMatrix,BigDensematrix,LittleDensematrix);
}

int* TaskPartition_Thread(SparseMatrixType SparseMatrix,int numthread)
{
  int row=SparseMatrix.outerSize();
  int col=SparseMatrix.innerSize();
  int nnz=SparseMatrix.nonZeros();
  int avg=(nnz+row)/numthread;
  int det=avg;
  int now=1;
  auto outIndexstart=SparseMatrix.outerIndexPtr();
  int *thread;
  size_t nSize=static_cast<size_t>(numthread+1);
  thread=new int[nSize];
  memset(thread,0,sizeof(int)*nSize);
  thread[0]=0;
  for(int i=1;i<numthread;i++){
    while(now<=row){
      if(outIndexstart[now]+now>=det){
        det=det+avg;
        break;
      }
      thread[i]=now;
      now=now+1;
    }
    if(thread[i]==0){
      thread[i]=now;
    }
  }
  thread[numthread]=row;
  return thread;
}

typedef struct{
  int startrow;
  int endrow;
  int startcol;
  int numcol;
  int col;
  int T_K;
  int T_W;
  float *Value;
  int *InnerIndex;
  int *OuterIndex;
  MatrixXf *result;
  MatrixXf *DenseMatrix;
  int tid;
}MY_ARGS;

//pthread_mutex_t mutex;

void* myfunc(void* args)
{
  MY_ARGS* p=(MY_ARGS*)args;
  cpu_set_t mask;
  pid_t pid=gettid();
  CPU_ZERO(&mask);
  CPU_SET(p->tid,&mask);
  int syscallret=sched_setaffinity(pid,sizeof(cpu_set_t),&mask);
  //printf("ID:%lu, CPU %d\n",p->tid,sched_getcpu());
  for(int H_o=(p->startrow);H_o<(p->endrow);H_o++){
    for(int W_o=(p->OuterIndex[H_o]);W_o<(p->OuterIndex[H_o+1]);W_o+=p->T_W){
      for(int K_o=0;K_o<(p->col);K_o+=p->T_K){
        int scol=std::min(p->T_W,p->OuterIndex[H_o+1]-W_o);
        int dcol=std::min(p->T_K,p->col-K_o);
        MatrixXf sm1(1,scol),dm1(scol,dcol),result1(1,dcol);
        for(int W_i=W_o;W_i<W_o+scol;W_i++){
          sm1(0,W_i-W_o)=(p->Value[W_i]);
          dm1.block(W_i-W_o,0,1,dcol)=(*(p->DenseMatrix)).block(p->InnerIndex[W_i],K_o,1,dcol);
        }
        result1.noalias()=sm1*dm1;
        //pthread_mutex_lock(&mutex);
        (*(p->result)).block(H_o,K_o+(p->startcol),1,dcol)+=result1;
        //pthread_mutex_unlock(&mutex);
        /*for(int W_i=W_o;(W_i<W_o+p->T_W)&&(W_i<(p->OuterIndex[H_o+1]));W_i++){
          for(int K_i=K_o;(K_i<K_o+p->T_K)&&(K_i<(p->col));K_i++){
            //std::cout<<H_o<<" "<<W_o<<" "<<K_o<<" "<<W_i<<" "<<K_i<<std::endl;
            p->result[H_o*(p->numcol)+(p->startcol)+K_i]+=p->Value[W_i]*(p->DenseMatrix[p->InnerIndex[W_i]*(p->col)+K_i]);
          }
        }*/
      }
    }
  }
  //printf("ID:%lu, CPU %d\n",p->tid,sched_getcpu());

  return nullptr;
}

bool Judgeresult(MatrixXf& result,int row,int col,MatrixXf& BigDenseMatrix,MatrixXf& LittleDenseMatrix,SparseMatrixType& SparseMatrix)
{
  bool flag=true;
  MatrixXf result_matrix(row,BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize());
  MatrixXf Densematrix(BigDenseMatrix.outerSize(),BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize());
  Densematrix.block(0,0,BigDenseMatrix.outerSize(),BigDenseMatrix.innerSize())=BigDenseMatrix;
  Densematrix.block(0,BigDenseMatrix.innerSize(),BigDenseMatrix.outerSize(),LittleDenseMatrix.innerSize())=LittleDenseMatrix;
  result_matrix.noalias()=SparseMatrix*Densematrix;
  for(int i=0;i<row;i++){
    for(int j=0;j<col;j++){
      if(fabs(result(i,j)-result_matrix(i,j))>eps){
        flag=false;
        std::cout<<i<<" "<<j<<std::endl;
      }
    }
  }
  return flag;
}

void compute();

void Process(std::string filename,std::string outname,float divisionratio,int Bigthreadnum,int Littlethreadnum){
  std::cout<<filename<<std::endl;
  //Eigen::initParallel();
  Originalmat Sparsemat,Densemat;
  int nnz;
  std::cout<<divisionratio<<::std::endl;
  std::tie(Sparsemat,Densemat,nnz)=ReadData(filename);
  int row=Sparsemat.get_row();
  int col=Sparsemat.get_col();
  float* mat=Sparsemat.get_mat();
  int *MappedRow=WeightReorder(Sparsemat);
  //init_Densematrix(Densemat);
  std::pair<Originalmat,Originalmat>SplitDensemat=DenseClusterPartition(Densemat,divisionratio);
  Originalmat BigDensemat=SplitDensemat.first,LittleDensemat=SplitDensemat.second;
  SparseMatrixType SparseMatrix;
  MatrixXf BigDenseMatrix,LittleDenseMatrix;
  std::tie(SparseMatrix,BigDenseMatrix,LittleDenseMatrix)=ConvertFormat(Sparsemat,BigDensemat,LittleDensemat,nnz);
  auto value=SparseMatrix.valuePtr();
  auto innerindex=SparseMatrix.innerIndexPtr();
  auto outerindex=SparseMatrix.outerIndexPtr();
  int *thread=TaskPartition_Thread(SparseMatrix,Littlethreadnum);
  size_t nSize=static_cast<size_t>(Bigthreadnum+Littlethreadnum);
  pthread_t* thread_handles=new pthread_t[nSize];
  MY_ARGS* args=new MY_ARGS[nSize];
  nSize=static_cast<size_t>(SparseMatrix.outerSize()*(BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize()));
  MatrixXf result(SparseMatrix.outerSize(),BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize());
  /*
  for(int i=0;i<=Littlethreadnum;i++){
    std::cout<<thread[i]<<" ";
  }
  std::cout<<std::endl;*/
  for(int i=0;i<Littlethreadnum;i++){
    args[i].startrow=thread[i];
    args[i].endrow=thread[i+1];
    args[i].InnerIndex=innerindex;
    args[i].OuterIndex=outerindex;
    args[i].Value=value;
    args[i].DenseMatrix=&LittleDenseMatrix;
    args[i].col=LittleDensemat.get_col();
    args[i].T_W=1;
    args[i].T_K=1;  
    args[i].startcol=BigDenseMatrix.innerSize();
    args[i].numcol=BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize();
    args[i].result=&result;
    args[i].tid=i;
  }
  delete []thread;
  thread=TaskPartition_Thread(SparseMatrix,Bigthreadnum);
  /*for(int i=0;i<=Bigthreadnum;i++){
    std::cout<<thread[i]<<" ";
  }
  std::cout<<std::endl;*/
  for(int i=0;i<Bigthreadnum;i++){
    args[i+Littlethreadnum].startrow=thread[i];
    args[i+Littlethreadnum].endrow=thread[i+1];
    args[i+Littlethreadnum].InnerIndex=innerindex;
    args[i+Littlethreadnum].OuterIndex=outerindex;
    args[i+Littlethreadnum].Value=value;
    args[i+Littlethreadnum].DenseMatrix=&BigDenseMatrix;
    args[i+Littlethreadnum].col=BigDensemat.get_col();
    args[i+Littlethreadnum].T_W=1;
    args[i+Littlethreadnum].T_K=1;  
    args[i+Littlethreadnum].startcol=0;
    args[i+Littlethreadnum].numcol=BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize();
    args[i+Littlethreadnum].result=&result;
    args[i+Littlethreadnum].tid=i+4;
  }
  int numberOFProcessors=sysconf(_SC_NPROCESSORS_CONF);
  //printf("Number of processors: %d\n",numberOFProcessors);
  int warmup=0,repeat=1;
  for(int k=0;k<warmup;k++){
    
    for(int i=0;i<Bigthreadnum+Littlethreadnum;i++){
      pthread_create(&thread_handles[i],NULL,myfunc,(void *)(&args[i]));
    }
    for(int i=0;i<Bigthreadnum+Littlethreadnum;i++){
        pthread_join(thread_handles[i],NULL);
    }
    
  }
  struct timeval start, end;
  struct timeval ComputeStart,ComputeEnd;
  double time=0;
  std::vector<double> times;
  gettimeofday(&ComputeStart,NULL);
  for(int k=0;k<repeat;k++){
    
    gettimeofday(&start,NULL);
    for(int i=0;i<Bigthreadnum+Littlethreadnum;i++){
      pthread_create(&thread_handles[i],NULL,myfunc,(void *)(&args[i]));
    }
    for(int i=0;i<Bigthreadnum+Littlethreadnum;i++){
      pthread_join(thread_handles[i],NULL);
    } 
    gettimeofday(&end,NULL);
    time=( end.tv_usec - start.tv_usec )/1000000.0  + (end.tv_sec - start.tv_sec);
    times.push_back(time);
    
  }
  gettimeofday(&ComputeEnd,NULL);
  
  row=SparseMatrix.outerSize();
  col=BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize();
  int flag=Judgeresult(result,row,col,BigDenseMatrix,LittleDenseMatrix,SparseMatrix);
  if(flag){
    std::cout<<"result correct"<<std::endl;
  }
  else{
    std::cout<<"result error"<<std::endl;
  }
  double sum=std::accumulate(std::begin(times),std::end(times),0.0);
  double mean=sum/(repeat*1.0);
  double variance=0.0;
  for(int i=0;i<repeat;i++){
    variance=variance+pow(times[i]-mean,2);
  }
  variance=variance/(repeat*1.0);
  std::ofstream out;
  out.open(outname,std::ios::out|std::ios::app);
  out<<filename<<',';
  out<<1.0-(nnz*1.0)/(row*col*1.0)<<',';
  out<<row<<',';
  out<<col<<',';
  out<<Densemat.get_col()<<',';
  out<<mean<<',';
  out<<variance<<',';
  out<<ComputeStart.tv_sec*1000+ComputeStart.tv_usec/1000<<',';
  out<<ComputeEnd.tv_sec*1000+ComputeEnd.tv_usec/1000<<std::endl;
  out.close();
  delete []MappedRow;
  delete []thread;
  delete []args;
  delete []thread_handles;
}

int main(int argc,char **argv){
  if(argc!=5){
    std::cout<<"please input correct filename and output document"<<std::endl;
  }
  std::string filenamepath=argv[1];
  std::string outname=argv[2];
  float divisionratio=1.0-1.0/(8.33333/5.17241+1.0);
  int Bigthreadnum=atoi(argv[3]);
  int Littlethreadnum=atoi(argv[4]);
  //std::cout<<Bigthreadnum<<" "<<Littlethreadnum<<std::endl;
  std::vector<std::string>filenames;
  std::ifstream read_file(filenamepath,std::ios::in);
  std::string name;
  while(read_file>>name){
    filenames.push_back(name);
  }
  read_file.close();
  int n=filenames.size();
  std::ofstream out;
  out.open(outname,std::ios::out|std::ios::app);
  out<<"filename"<<','<<"sparse rate"<<','<<"H"<<','<<"W"<<','<<"N"<<','<<"Sparse_time_average"<<','<<"Sparse_time_val"<<','<<"S_s"<<','<<"S_e"<<std::endl;
  out.close();
  for(int i=0;i<n;i++){
    std::string filename=filenames[i];
    Process(filename,outname,divisionratio,Bigthreadnum,Littlethreadnum);
  }
}