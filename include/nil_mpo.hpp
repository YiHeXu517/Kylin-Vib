/* MPO class with dense tensor */

#pragma once

#include "dense_mps.h"

namespace KylinVib
{
    template<typename T> class MPO : public vector<Tensor<T,4>>
    {
        public:
        MPO() = default;
        MPO(MPO<T> const & r)
        : vector<Tensor<T,4>>(r)
        {

        }
        MPO(MPO<T> && r)
        : vector<Tensor<T,4>>(move(r))
        {

        }
        MPO(size_t ns)
        : vector<Tensor<T,4>>(ns)
        {

        }
        ~MPO() = default;
        MPO<T> & operator=(MPO<T> const & r)
        {
            vector<Tensor<T,4>>::operator=(r);
            return *this;
        }
        MPO<T> & operator=(MPO<T> && r)
        {
            vector<Tensor<T,4>>::operator=(r);
            return *this;
        }
        MPO<T> & operator+=(MPO<T> const & r)
        {
            size_t ns = this->size();
            (*this)[0] = stack<T,4>((*this)[0],r[0],{3});
            (*this)[ns-1] = stack<T,4>((*this)[ns-1],r[ns-1],{0});
            for(size_t i=1;i<ns-1;++i)
            {
                (*this)[i] = stack<T,4>((*this)[i],r[i],{0,3});
            }
            return *this;
        }
        MPO<T> operator+(MPO<T> const & r) const
        {
            MPO<T> res(*this);
            res += r;
            return res;
        }
        MPO<T> & operator*=(double val)
        {
            (*this)[0] *= val;
            return *this;
        }
        MPO<T> operator*(T const & val) const
        {
            MPO<T> res(*this);
            res[0] = res[0] * val;
            return res;
        }
        void print(double tol = 1e-14) const
        {
            for(size_t site=0;site<this->size();++site)
            {
                cout << "Site: " << site+1 << endl;
                (*this)[site].print(tol);
            }
        }
        void canon(double tol = 1e-8, size_t maxdim = 1000)
        {
            size_t ns = this->size();
            for(size_t i=0;i<ns-1;++i)
            {
                auto[lef,rig] = svd<T,3,1>((*this)[i],'r',tol,maxdim,'r');
                (*this)[i] = move(lef);
                (*this)[i+1] = prod<T,2,4,1>(rig,(*this)[i+1],{1},{0});
            }
            for(size_t i=ns-1;i>0;--i)
            {
                auto[lef,rig] = svd<T,1,3>((*this)[i],'l',tol,maxdim,'r');
                (*this)[i] = move(rig);
                (*this)[i-1] = prod<T,4,2,1>((*this)[i-1],lef,{3},{0});
            }
        }
        T join(MPS<T> const & s1, MPS<T> const & s2) const
        {
            size_t ns = this->size();
            Tensor<T,3> env({1,1,1});
            env.ptr()[0] = 1.0;
            for(size_t i=0;i<ns;++i)
            {
                env = sweep(env,(*this)[i],s1[i],s2[i]);
            }
            return env.ptr()[0];
        }
        MPS<T> apply_op(MPS<T> const & s, double tol = 1e-14, size_t maxdim = 1000) const
        {
            size_t ns = this->size();
            Tensor<double,3> env({1,1,1});
            env.ptr()[0] = 1.0;
            MPS<T> res(ns);
            for(size_t i=0;i<ns;++i)
            {
                Tensor<T,4> lss = prod<T,3,3,1>(env,s[i],{2},{0});
                Tensor<T,4> lso = prod<T,4,4,2>(lss,(*this)[i],{1,2},{0,2});
                lso = transpose<T,4>(lso,{0,2,3,1});
                if(i!=ns-1)
                {
                    auto[lef,rig] = svd<T,2,2>(lso,'r',tol,maxdim);
                    res[i] = move(lef);
                    env = move(rig);
                }
                else
                {
                    Tensor<T,3> lsoc({lso.shape()[0],lso.shape()[1],lso.shape()[2]});
                    #pragma omp parallel for
                    for(size_t j=0;j<lsoc.size();++j)
                    {
                        lsoc.ptr()[j] = lso.ptr()[j];
                    }
                    res[i] = move(lsoc);
                }
            }
            for(size_t i=ns-1;i>0;--i)
            {
                auto[lef,rig] = svd<T,1,2>(res[i],'l',tol,maxdim);
                res[i] = move(rig);
                res[i-1] = prod<T,3,2,1>(res[i-1],lef,{2},{0});
            }
            return res;
        }
        // make diagonal 
        MPS<T> diag_state() const
        {
            size_t ns = this->size();
            MPS<T> res(ns);
            for(size_t i=0;i<ns;++i)
            {
                size_t bl = (*this)[i].shape()[0], br = (*this)[i].shape()[3],
                d = (*this)[i].shape()[1];
                Tensor<T,3> si({bl,d,br});
                for(size_t j=0;j<si.data.size();++j)
                {
                    array<size_t,3> sidx = si.make_indices(j);
                    si.ptr()[j] = (*this)[i]({sidx[0],sidx[1],sidx[1],sidx[2]});
                }
                res[i] = si;
            }
            res.canon();
            return res;
        }
        // product tools
        static Tensor<T,3> sweep(Tensor<T,3> const & env, Tensor<T,4> const & mpo,
        Tensor<T,3> const & mps1, Tensor<T,3> const & mps2, char l2r = 'r')
        {
            if(l2r=='r')
            {
                Tensor<T,4> lss = prod<T,3,3,1>(env,mps2,{2},{0});
                Tensor<T,4> lso = prod<T,4,4,2>(lss,mpo,{1,2},{0,2});
                Tensor<T,3> res = prod<T,3,4,2>(conj<T,3>(mps1),lso,{0,1},{0,2});
                return transpose<T,3>(res,{0,2,1});
            }
            else
            {
                Tensor<T,4> lss = prod<T,3,3,1>(mps2,env,{2},{2});
                Tensor<T,4> lso = prod<T,4,4,2>(mpo,lss,{2,3},{1,3});
                Tensor<T,3> res = prod<T,4,3,2>(lso,conj<T,3>(mps1),{1,3},{1,2});
                return transpose<T,3>(res,{2,0,1});
            }
        }
    };
}
