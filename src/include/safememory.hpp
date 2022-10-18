#include<memory>
#include<cstring>

template<typename _T>
class Ptr : public std::shared_ptr<_T>
{
public:

private:

};

template<typename _T>
inline void deleteSafe(_T*& p) {
    if (nullptr == p)
        return;
    delete p;
    p = nullptr;
}

template<typename _T>
inline void deleteArraySafe(_T*& p) {
    if (nullptr == p)
        return;
    delete[] p;
    p = nullptr;
}
