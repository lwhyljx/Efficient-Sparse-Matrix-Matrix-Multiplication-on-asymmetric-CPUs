#include "Originalmat.h"
#include <Eigen/Sparse>
#include <Eigen/Dense>


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
#include<pthread.h>

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
  //delete [] A;
  //delete []B;
  //B=nullptr;
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
  //delete []M;
}

std::pair<Originalmat,Originalmat> DenseClusterPartition(Originalmat& Densemat,float radio)
{
  int row,col;
  float* M;
  row=Densemat.get_row();
  col=Densemat.get_col();
  M=Densemat.get_mat();
  int colSplit=(int) round(radio*col);
  float *BigM,*LittleM;
  size_t BigSize=static_cast<size_t>(row)*static_cast<size_t>(colSplit);
  size_t LittleSize=static_cast<size_t>(row)*static_cast<size_t>(col-colSplit);
  BigM=new float[BigSize];
  LittleM=new float[LittleSize];
  for(int i=0;i<row;i++){
    memcpy(BigM+i*colSplit,M+i*col,sizeof(float)*colSplit);
    memcpy(LittleM+i*(col-colSplit),M+i*col+colSplit,sizeof(float)*(col-colSplit));
  }
  Originalmat BigDensemat,LittleDensemat;
  BigDensemat.setmat(row,colSplit,BigM);
  LittleDensemat.setmat(row,col-colSplit,LittleM);
  delete []BigM;
  delete []LittleM;
  return std::make_pair(BigDensemat,LittleDensemat);
}

std::tuple<SparseMatrixType,MatrixXf,MatrixXf> ConvertFormat(Originalmat& Sparsemat,Originalmat& BigDensemat,Originalmat& LittleDensemat,int nnz)
{
  int Sparsematrow=Sparsemat.get_row(),Sparsematcol=Sparsemat.get_col();
  int BigDensematrow=BigDensemat.get_row(),BigDensematcol=BigDensemat.get_col();
  int LittleDensematrow=LittleDensemat.get_row(),LittleDensematcol=LittleDensemat.get_col();
  MatrixXf BigDensematrix(BigDensematrow,BigDensematcol),LittleDensematrix(LittleDensematrow,LittleDensematcol);
  float* M=BigDensemat.get_mat();
  for(int i=0;i<BigDensematrow;i++){
    for(int j=0;j<BigDensematcol;j++){
      BigDensematrix(i,j)=M[i*BigDensematcol+j];
    }
  }
  M=LittleDensemat.get_mat();
  for(int i=0;i<LittleDensematrow;i++){
    for(int j=0;j<LittleDensematcol;j++){
      LittleDensematrix(i,j)=M[i*LittleDensematcol+j];
    }
  }
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
  memset(thread,0,sizeof(thread));
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

float* result;

typedef struct{
  int *InnerIndex;
  int *OuterIndex;
  int startrow;
  int endrow;
  int startcol;
  int numcol;
  int T_K;
  int T_W;
  float *Value;
  MatrixXf DenseMatrix;
}MY_ARGS;

void* myfunc(void* args)
{
  MY_ARGS* p=(MY_ARGS*)args;
  int col=(p->DenseMatrix).innerSize();
  for(int H_o=(p->startrow);H_o<(p->endrow);H_o++){
    for(int W_o=(p->OuterIndex[H_o]);W_o<(p->OuterIndex[H_o+1]);W_o+=p->T_W){
      for(int K_o=0;K_o<col;K_o+=p->T_K){
        for(int W_i=W_o;(W_i<W_o+p->T_W)&&(W_i<(p->OuterIndex[H_o+1]));W_i++){
          for(int K_i=K_o;(K_i<K_o+p->T_K)&&(K_i<col);K_i++){
            result[H_o*(p->numcol)+p->startcol+K_i]+=p->Value[W_i]*(p->DenseMatrix(p->InnerIndex[W_i],K_i));
          }
        }
      }
    }
  }
  return nullptr;
}

bool Judgeresult(int row,int col,MatrixXf& BigDenseMatrix,MatrixXf& LittleDenseMatrix,SparseMatrixType& SparseMatrix)
{
  bool flag=true;
  MatrixXf result_matrix1(row,BigDenseMatrix.innerSize());
  result_matrix1.noalias()=SparseMatrix*BigDenseMatrix;
  MatrixXf result_matrix2(row,LittleDenseMatrix.innerSize());
  result_matrix2.noalias()=SparseMatrix*LittleDenseMatrix;
  int Bigcol=BigDenseMatrix.innerSize(),Littlecol=LittleDenseMatrix.innerSize();
  for(int i=0;i<row;i++){
    for(int j=0;j<Bigcol;j++){
      if(std::fabs(result[i*col+j]-result_matrix1(i,j))>eps){
        //std::cout<<result[i*col+j]<<" "<<result_matrix1(i,j)<<std::endl;
        flag=false;
      }
    }
    for(int j=Bigcol;j<col;j++){
      if(std::fabs(result[i*col+j]-result_matrix2(i,j-Bigcol))>eps){
        //std::cout<<i<<" "<<j<<std::endl;
        flag=false;
      }
    }
  }
  return flag;
}

int main(int argc,char **argv){
  std::string filename="/home/ljx/Efficient-Sparse-Matrix-Matrix-Multiplication-on-asymmetric-CPUs/data/0.8/1.smt";
  Originalmat Sparsemat,Densemat;
  int nnz;
  std::tie(Sparsemat,Densemat,nnz)=ReadData(filename);
  int row=Sparsemat.get_row();
  int col=Sparsemat.get_col();
  float* mat=Sparsemat.get_mat();
  int *MappedRow=WeightReorder(Sparsemat);
  init_Densematrix(Densemat);
  std::pair<Originalmat,Originalmat>SplitDensemat=DenseClusterPartition(Densemat,0.6);
  Originalmat BigDensemat=SplitDensemat.first,LittleDensemat=SplitDensemat.second;
  SparseMatrixType SparseMatrix;
  MatrixXf BigDenseMatrix,LittleDenseMatrix;
  std::tie(SparseMatrix,BigDenseMatrix,LittleDenseMatrix)=ConvertFormat(Sparsemat,BigDensemat,LittleDensemat,nnz);
  auto value=SparseMatrix.valuePtr();
  auto innerindex=SparseMatrix.innerIndexPtr();
  auto outerindex=SparseMatrix.outerIndexPtr();
  int numthread=4;
  int *thread=TaskPartition_Thread(SparseMatrix,numthread);
  size_t nSize=static_cast<size_t>(numthread*2+1);
  pthread_t* thread_handles=new pthread_t[nSize];
  MY_ARGS* args=new MY_ARGS[nSize];
  nSize=static_cast<size_t>(SparseMatrix.outerSize()*(BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize()));
  result=new float[nSize];
  memset(result,0,sizeof(result));
  for(int i=0;i<numthread;i++){
    args[i].startrow=args[i+numthread].startrow=thread[i];
    args[i].endrow=args[i+numthread].endrow=thread[i+1];
    args[i].InnerIndex=args[i+numthread].InnerIndex=innerindex;
    args[i].OuterIndex=args[i+numthread].OuterIndex=outerindex;
    args[i].Value=args[i+numthread].Value=value;
    args[i].DenseMatrix=BigDenseMatrix;
    args[i+numthread].DenseMatrix=LittleDenseMatrix;
    args[i].T_W=2;
    args[i].T_K=4;
    args[i+numthread].T_W=1;
    args[i+numthread].T_K=2;  
    args[i].startcol=0;
    args[i].numcol=BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize();
    args[i+numthread].startcol=BigDenseMatrix.innerSize();
    args[i+numthread].numcol=BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize();
    std::cout<<args[i].startrow<<" "<<args[i+numthread].startrow<<std::endl;
    std::cout<<args[i].endrow<<" "<<args[i+numthread].endrow<<std::endl;
  }
  for(int i=0;i<numthread*2;i++){
    if(pthread_create(&thread_handles[i],NULL,myfunc,(void *)(&args[i]))){
      std::cout<<"error"<<std::endl;
    }
  }
  std::cout<<BigDenseMatrix.innerSize()<<std::endl;
  for(int i=0;i<numthread*2;i++){
    if(pthread_join(thread_handles[i],NULL)){
      std::cout<<i<<std::endl;
    }
  }
  row=SparseMatrix.outerSize();
  col=BigDenseMatrix.innerSize()+LittleDenseMatrix.innerSize();
  int flag=Judgeresult(row,col,BigDenseMatrix,LittleDenseMatrix,SparseMatrix);
  if(flag){
    std::cout<<"result correct"<<std::endl;
  }
  else{
    std::cout<<"result error"<<std::endl;
  }
}