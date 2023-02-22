#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdlib>


template<class T>
class dev_array
{
    
public:
    explicit dev_array()
        : start_(0),
          end_(0)
    {}

    explicit dev_array(size_t size)
    {
        allocate(size);
    }

    ~dev_array()
    {
        free();
    }

    void resize(size_t size)
    {
        free();
        allocate(size);
    }

    size_t getSize() const
    {
        return end_ - start_;
    }

    const T* getData() const
    {
        return start_;
    }

    T* getData()
    {
        return start_;
    }

    void set(const T* src,size_t size)
    {
        size_t min=std::min(size,getSize());
        memcpy(start_, src,min*sizeof(T));
    }

    void get(T *dest,size_t size)
    {
        size_t min=std::min(size,getSize());
        memcpy(dest,start_,min*sizeof(T));
    }

private:
    void allocate(size_t size)
    {
        start_=(T*) malloc(size*sizeof(T));
        end_=start_+size;
    }

    void free()
    {
        if(start_!=0)
        {
            std::free(start_);
            start_=end_=0;
        }
    }

    T* start_;
    T* end_;
};

#endif