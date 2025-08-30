/* MPS class with dense tensor */

#pragma once
#include "nil_linalg.hpp"

namespace Nil
{
    template<typename T> class MPS : public  vector<Tensor<T,3>>
    {
        public:
        MPS() = default;
        MPS(MPS<T> const & r)
        :  vector<Tensor<T,3>>(r)
        {

        }
        MPS(MPS<T> && r)
        :  vector<Tensor<T,3>>( move(r))
        {

        }
        MPS(size_t ns)
        :  vector<Tensor<T,3>>(ns)
        {

        }
        MPS(size_t ns, size_t nphys, map<size_t,size_t> const & cfg)
        :  vector<Tensor<T,3>>(ns)
        {
            for(size_t i=0;i<ns;++i)
            {
                Tensor<T,3> si({1,nphys,1});
                (*this)[i] = si;
            }
            for(const auto & [key,value] : cfg)
            {
                (*this)[key]({0,value,0}) = 1.0;
            }
        }
        MPS(size_t ns, size_t nphys, vector<size_t> const & cfg)
        :  vector<Tensor<T,3>>(ns)
        {
            for(size_t i=0;i<ns;++i)
            {
                Tensor<T,3> si({1,nphys,1});
                si({0,cfg[i],0}) = 1.0;
                (*this)[i] = si;
            }
        }
        MPS(size_t ns, vector<size_t> const & nphys, vector<size_t> const & cfg)
        :  vector<Tensor<T,3>>(ns)
        {
            for(size_t i=0;i<ns;++i)
            {
                Tensor<T,3> si({1,nphys[i],1});
                si({0,cfg[i],0}) = 1.0;
                (*this)[i] = si;
            }
        }
        MPS(size_t ns, size_t nphys, size_t bd)
        :  vector<Tensor<T,3>>(ns)
        {
            Tensor<T,3> s1({1,nphys,bd});
            s1.rand_fill();
            (*this)[0] = s1;
            Tensor<T,3> s2({bd,nphys,1});
            s2.rand_fill();
            (*this)[ns-1] = s2;
            for(size_t i=1;i<ns-1;++i)
            {
                Tensor<T,3> si({bd,nphys,bd});
                si.rand_fill();
                (*this)[i] = si;
            }
        }
        ~MPS() = default;
        MPS<T> & operator=(MPS<T> const & r)
        {
             vector<Tensor<T,3>>::operator=(r);
            return *this;
        }
        MPS<T> & operator=(MPS<T> && r)
        {
             vector<Tensor<T,3>>::operator=(r);
            return *this;
        }
        MPS<T> & operator+=(MPS<T> const & r)
        {
            size_t ns = this->size();
            (*this)[0] = stack<T,3>((*this)[0],r[0],{2});
            (*this)[ns-1] = stack<T,3>((*this)[ns-1],r[ns-1],{0});
            for(size_t i=1;i<ns-1;++i)
            {
                (*this)[i] = stack<T,3>((*this)[i],r[i],{0,2});
            }
            return *this;
        }
        MPS<T> operator+(MPS<T> const & r) const
        {
            MPS<T> res(*this);
            res += r;
            return res;
        }
        MPS<T> & operator*=(double val)
        {
            (*this)[0] *= val;
            return *this;
        } 
        MPS<T> operator*(T const & val) const
        {
            MPS<T> res(*this);
            res[0] = res[0] * val;
            return res;
        }
        void prsize_t(double tol = 1e-14) const
        {
            for(size_t site=0;site<this->size();++site)
            {
                 cout << "Site: " << site+1 <<  endl;
                (*this)[site].print(tol);
            }
        }
        void canon(double tol = 1e-14, size_t maxdim = 1000)
        {
            size_t ns = this->size();
            for(size_t i=0;i<ns-1;++i)
            {
                auto[lef,rig] = svd<T,2,1>((*this)[i],'r',tol,maxdim);
                (*this)[i] =  move(lef);
                (*this)[i+1] = prod<T,2,3,1>(rig,(*this)[i+1],{1},{0});
            }
            for(size_t i=ns-1;i>0;--i)
            {
                auto[lef,rig] = svd<T,1,2>((*this)[i],'l',tol,maxdim);
                (*this)[i] =  move(rig);
                (*this)[i-1] = prod<T,3,2,1>((*this)[i-1],lef,{2},{0});
            }
        }
        T overlap(MPS<T> const & s) const
        {
            size_t ns = this->size();
            Tensor<T,2> env({1,1});
            env.data[0] = 1.0;
            for(size_t i=0;i<ns;++i)
            {
                env = sweep(env,(*this)[i],s[i]);
            }
            return env.data[0];
        }
        // extract dominant cfgs
        vector<LabArr<T,2>> dominant(size_t nst) const
        {
             vector<LabArr<T,2>> levs = all_levels((*this)[0]), kepts;
            kepts = truncate_levels(levs,nst);
            for(size_t site=1;site<this->size();++site)
            {
                levs = all_levels((*this)[site]);
                vector<LabArr<T,2>> envs(levs.data.size()*kepts.data.size());
                #pragma omp parallel for
                for(size_t i=0;i<envs.size();++i)
                {
                    size_t i1 = i / levs.size(), i2 = i % levs.size();
                    envs[i] = mat_prod<T>(kepts[i1],levs[i2]);
                }
                kepts = truncate_levels(envs,nst);
            }
            return kepts;
        }
        // product tools
        static Tensor<T,2> sweep(Tensor<T,2> const & env, Tensor<T,3> const & mps1, 
        Tensor<T,3> const & mps2, char l2r = 'r')
        {
            if(l2r=='r')
            {
                Tensor<T,3> ls2 = prod<T,2,3,1>(env,mps2,{1},{0});
                Tensor<T,2> res = prod<T,3,3,2>(conj<T,3>(mps1),ls2,{0,1},{0,1});
                return res;
            }
            else
            {
                Tensor<T,3> ls2 = prod<T,3,2,1>(mps2,env,{2},{1});
                Tensor<T,2> res = prod<T,3,3,2>(conj<T,3>(mps1),ls2,{1,2},{1,2});
                return res;
            }
        }
        // two-site overlap
        static Tensor<T,4> make_overlap(Tensor<T,2> const & envl, Tensor<T,3> const & mps1, 
        Tensor<T,3> const & mps2, Tensor<T,2> const & envr)
        {
            Tensor<T,3> ls1 = prod<T,2,3,1>(envl,mps1,{1},{0});
            Tensor<T,3> rs2 = prod<T,3,2,1>(mps2,envr,{2},{1});
            Tensor<T,4> res = prod<T,3,3,1>(ls1,rs2,{2},{0});
            return res;
        } 
        static vector<LabArr<T,2>> all_levels(Tensor<T,3> const & mps)
        {
            vector<LabArr<T,2>> res(mps.shape[1],{mps.shape[0],mps.shape[2]});
            #pragma omp parallel for
            for(size_t i=0;i<mps.size();++i)
            {
                array<size_t,3> sidx = mps.make_indices(i);
                res[sidx[1]]({sidx[0],sidx[2]}) = mps[i];
            }
            size_t d = res.size();
            for(size_t i=0;i<d;++i)
            {
                res[i].labs.push_back(i);
            }
            return res;
        }
        static vector<LabArr<T,2>> truncate_levels( vector<LabArr<T,2>> const & r, size_t nst)
        {
            if(r.size()<=nst)
            {
                vector<LabArr<T,2>> res(r);
                return res;
            }
            vector< tuple<size_t,double>> tups;
            for(size_t i=0;i<r.size();++i)
            {
                tups.push_back(  make_tuple(i,r[i].norm()) );
            }
            sort(tups.begin(),tups.end(),
            [&tups]( tuple<size_t,double> x1,  tuple<size_t,double> x2){
            return  get<1>(x1)> get<1>(x2);});
            vector<LabArr<T,2>> res(nst);
            #pragma omp parallel for
            for(size_t i=0;i<nst;++i)
            {
                res[i] = r[ get<0>(tups[i])];
            }
            return res;
        }
    };
}
