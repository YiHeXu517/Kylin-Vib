#pragma once
#include "util.hpp"
namespace Nil
{
    template<typename T, size_t N> struct Tensor
    {
        array<size_t,N> shape;
        array<size_t,N> dist;
        vector<T> data;

        Tensor() = default;
        ~Tensor() = default;
        Tensor(Tensor<T,N> const & r)
        :shape(r.shape),dist(r.dist),data(r.data)
        {

        }
        Tensor(Tensor<T,N> && r)
        :shape(move(r.shape)),dist(move(r.dist)),data(move(r.data))
        {

        }
        Tensor(array<size_t,N> const & other_shape)
        :shape(other_shape),data(accumulate(other_shape.begin(),other_shape.end(),1
        ,multiplies<size_t>()))
        {
            dist[N-1] = 1;
            for(size_t ax=1;ax<N;++ax)
            {
                dist[N-1-ax] = dist[N-ax] * shape[N-ax];
            }
        }
        Tensor(initializer_list<size_t> other_shape)
        :data(accumulate(other_shape.begin(),other_shape.end(),1,multiplies<size_t>()))
        {
            copy(other_shape.begin(),other_shape.end(),shape.begin());
            dist[N-1] = 1;
            for(size_t ax=1;ax<N;++ax)
            {
                dist[N-1-ax] = dist[N-ax] * shape[N-ax];
            }
        }
        Tensor<T,N> & operator=(Tensor<T,N> const & r)
        {
            shape = r.shape;
            dist = r.dist;
            data = r.data;
            return *this;
        }
        Tensor<T,N> & operator=(Tensor<T,N> && r)
        {
            shape = move(r.shape);
            dist = move(r.dist);
            data = move(r.data);
            return *this;
        }
        T & operator()(array<size_t,N> const & idx)
        {
            size_t pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
            return data[pos];
        }
        T & operator()(initializer_list<size_t> idx)
        {
            size_t pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
            return data[pos];
        }
        T const & operator()(array<size_t,N> const & idx) const
        {
            size_t pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
            return data[pos];
        }
        T const & operator()(initializer_list<size_t> idx) const
        {
            size_t pos = inner_product(idx.begin(),idx.end(),dist.begin(),0);
            return data[pos];
        }
        Tensor<T,N> & operator+=(Tensor<T,N> const & r)
        {
            T ones(1.0);
            size_t data_size = data.size();
            if constexpr(is_same<T,double>::value)
            {
                cblas_daxpy(data_size,ones,r.data.data(),1,data.data(),1);
            }
            else
            {
                cblas_zaxpy(data_size,&ones,r.data.data(),1,data.data(),1);
            }
            return *this;
        }
        Tensor<T,N> operator+(Tensor<T,N> const & r)
        {
            Tensor<T,N> res(*this);
            res += r;
            return res;
        }
        Tensor<T,N> & operator*=(double val)
        {
            size_t data_size = data.size();
            if constexpr(is_same<T,double>::value)
            {
                cblas_dscal(data_size,val,data.data(),1);
            }
            else
            {
                cblas_zdscal(data_size,val,data.data(),1);
            }
            return *this;
        }
        Tensor<T,N> operator*(T const & val)
        {
            size_t data_size = data.size();
            Tensor<T,N> res(*this);
            if constexpr(is_same<T,double>::value)
            {
                cblas_dscal(data_size,val,res.data.data(),1);
            }
            else
            {
                cblas_zscal(data_size,&val,res.data.data(),1);
            }
            return res;
        }
        void print(double tol=1e-15) const
        {
            cout << "Shape:";
            cout << shape << endl;
            size_t data_size = data.size();
            for(size_t idx=0;idx<data_size;++idx)
            {
                array<size_t,N> idices;
                if(abs(data[idx])<=tol)
                {
                    continue;
                }
                for(size_t ax=0;ax<N;++ax)
                {
                    idices[ax] = idx / dist[ax] % shape[ax];
                }
                cout << idices << " | " << std::scientific << data[idx] << endl;
            }
        }
        double norm() const
        {
            double res = 0.0;
            size_t data_size = data.size();
            if constexpr(is_same<T,double>::value)
            {
                res = cblas_dnrm2(data_size,data.data(),1);
            }
            else
            {
                res = cblas_dznrm2(data_size,data.data(),1);
            }
            return res;
        }
        T overlap(Tensor<T,N> const & r) const
        {
            T res(0.0);
            size_t data_size = data.size();
            if constexpr(is_same<T,double>::value)
            {
                res = cblas_dnrm2(data_size,res.data.data(),1);
            }
            else
            {
                res = cblas_dznrm2(data_size,res.data.data(),1);
            }
            return res;
        }
    };
}
