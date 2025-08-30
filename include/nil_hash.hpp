#pragma once
#include "nil_tensor.hpp"
namespace Nil
{
    template<typename T, size_t N> struct LabArr : public Tensor<T,N>
    {
        vector<size_t> labs;

        LabArr() = default;
        LabArr(LabArr<T,N> const & r) : labs(r.labs),
        Tensor<T,N>(r)
        {

        }
        LabArr(LabArr<T,N> && r) : labs(move(r.labs)),
        Tensor<T,N>(move(r))
        {
            
        }
        LabArr(initializer_list<size_t> Sp)
        :Tensor<T,N>(Sp)
        {

        }
        ~LabArr() = default;
        LabArr<T,N> & operator=(LabArr<T,N> const & r)
        {
            labs = r.labs;
            Tensor<T,N>::operator=(r);
            return *this;
        }
        LabArr<T,N> & operator=(LabArr<T,N> && r)
        {
            labs = move(r.labs);
            Tensor<T,N>::operator=(move(r));
            return *this;
        }
        void print(double tol = 1e-15) const
        {
            cout << "Labels:[";
            for(size_t i=0;i<labs.size()-1;++i)
            {
                cout << labs[i] << ",";
            }
            cout << labs.back() << "]" << endl;
            Tensor<T,N>::print(tol);
        }
        void print_special(double EShift) const
        {
            cout << "Labels:[";
            for(size_t i=0;i<labs.size()-1;++i)
            {
                cout << labs[i] << ",";
            }
            cout << labs.back() << "] | " 
            <<    this->ptr()[0]+EShift << endl;
        }
    };
    
}