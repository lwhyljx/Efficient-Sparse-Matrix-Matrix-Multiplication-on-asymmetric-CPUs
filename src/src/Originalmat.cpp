#include "Originalmat.h"
#include "safememory.hpp"


#include<iostream>
#include<memory>
#include<cstring>
Originalmat::Originalmat():
      row(0),
      col(0),
      mat(nullptr){

      }

Originalmat::Originalmat(int rows,int cols)
{
    this->create(rows,cols);
}

Originalmat::Originalmat(const Originalmat& M)
{
  this->create(M.row,M.col);
  memcpy(mat,M.mat,M.row*M.col*sizeof(float));
}

Originalmat::~Originalmat()
{
  this->release();
}

int Originalmat::get_row() const
{
  return row;
}

int Originalmat::get_col() const
{
  return col;
}

float* Originalmat::get_mat() const
{
  return mat;
}

void Originalmat::setmat(int rows,int cols,float* M)
{
  this->release();
  this->create(rows,cols);
  memcpy(this->mat,M,rows*cols*sizeof(float));
}

void Originalmat::create(int rows,int cols)
{
  row=rows;
  col=cols;
  size_t nSize=static_cast<size_t>(row)*static_cast<size_t>(col);
  mat=new float[nSize];
}
void Originalmat::release()
{
  deleteArraySafe(mat);
  row=col=0;
}
Originalmat& Originalmat::operator =(const Originalmat& M)
{
  if(this!=&M){
    if(M.col!=col||M.row!=row){
      this->release();
      this->create(M.row,M.col);
      memcpy(mat,M.mat,row*col*sizeof(float));
    }
  }
  return *this;
}