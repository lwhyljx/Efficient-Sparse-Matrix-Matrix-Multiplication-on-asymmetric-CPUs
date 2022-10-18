#include<iostream>


class Originalmat
{
  public:
      Originalmat();

      Originalmat(int rows,int cols);

      Originalmat(const Originalmat& M);

      virtual ~Originalmat();

      int get_row() const;

      int get_col() const;

      float* get_mat() const;

      void setmat(int rows,int cols,float* M);

      void create(int rows,int cols);

      void release();

      Originalmat& operator =(const Originalmat& M);
  private:
      int row,col;
      float* mat;     
};
