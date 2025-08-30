/* DMRG implementation */

#pragma once

#include <deque>
#include "dense_mpo.h"
//#include "timer.h"

namespace KylinVib
{
    template<typename T> class DMRG
    {
        public:
        DMRG(MPO<T> const & op, MPS<T> const & s, double tol = 1e-8, size_t maxdim = 20)
        : ham_(op), state_(s), tol_(tol), MaxBond_(maxdim)
        {
            Tensor<T,3> edges({1,1,1});
            edges({0,0,0})= 1.0;
            envl_.push_back(edges);
            envr_.push_back(edges);
            size_t ns = op.size();
            for(size_t i=ns-1;i>1;--i)
            {
                envr_.push_back(MPO<T>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
            }
        }
        ~DMRG() = default;

        // Hv
        static Tensor<T,3> one_site_apply(Tensor<T,5> const & vlo, Tensor<T,3> const & s, Tensor<T,3> const & envr)
        {
            Tensor<T,4> los = prod<T,5,3,2>(vlo,s,{3,4},{0,1});
            Tensor<T,3> res = prod<T,4,3,2>(los,envr,{2,3},{1,2});
            return res;
        }
        static Tensor<T,4> two_site_apply(Tensor<T,5> const & vlo, Tensor<T,4> const & s2, Tensor<T,5> const & vro)
        {
            Tensor<T,5> los = prod<T,5,4,2>(vlo,s2,{3,4},{0,1});
            Tensor<T,4> res = prod<T,5,5,3>(los,vro,{2,3,4},{0,1,2});
            return res;
        }
        Tensor<T,4> two_site_lanczos(Tensor<T,4> const & x0, size_t site, T const & dt, size_t MicroIter)
        {
            vector<Tensor<T,4>> Vk(MicroIter,x0.shape());
            Tensor<T,5> vlo = prod<T,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<T,5>(vlo,{0,2,4,1,3});
            Tensor<T,5> vro = prod<T,4,3,1>(ham_[site+1],envr_.back(),{3},{1});
            vro = transpose<T,5>(vro,{0,2,4,1,3});
            T beta = x0.norm();
            Vk[0] = x0 / beta;
            vector<tuple<size_t,size_t,T>> HmTrips;
            for(size_t i=0;i<MicroIter;++i)
            {
                Tensor<T,4> hv = two_site_apply(vlo,Vk[i],vro) * dt;
                for(size_t j=0;j<i+1;++j)
                {
                    T ov = -1.0 * Vk[j].overlap(hv);
                    HmTrips.push_back(make_tuple(j,i,-1.0*ov));
                    hv += Vk[j] * ov;
                }
                T nmhv = hv.norm();
                HmTrips.push_back(make_tuple(i+1,i,nmhv));
                if(i!=MicroIter-1)
                {
                    hv = hv / nmhv;
                    Vk[i+1] = move(hv);
                }
            }
            size_t Nm = Vk.size();
            Tensor<T,2> Hm({Nm,Nm}),eaHm;
            for(auto const & tp : HmTrips)
            {
                Hm({get<0>(tp),get<1>(tp)}) = get<2>(tp);
            }
            eaHm = eig<T>(Hm);
            for(size_t i=0;i<MicroIter;++i)
            {
                eaHm({i,i}) = exp(eaHm({i,i}));
            }
            eaHm = prod<T,2,2,1>(Hm,eaHm);
            eaHm = prod<T,2,2,1>(eaHm,conj<T,2>(Hm),{1},{1});
            Tensor<T,3> res = Vk[0] * eaHm({0,0});
            for(size_t i=1;i<MicroIter;++i)
            {
                res += Vk[i] * eaHm({i,0});
            }
            res =  res * beta;
            return res;
        }
        Tensor<T,3> one_site_lanczos(Tensor<T,3> const & x0, size_t site, T const & dt, size_t MicroIter)
        {
            Tensor<T,5> vlo = prod<T,3,4,1>(envl_.back(),ham_[site],{1},{0});
            vlo = transpose<T,5>(vlo,{0,2,4,1,3});
            vector<Tensor<T,3>> Vk(MicroIter,x0.shape());
            T beta = x0.norm();
            Vk[0] = x0 / beta;
            vector<tuple<size_t,size_t,T>> HmTrips;
            for(size_t i=0;i<MicroIter;++i)
            {
                Tensor<T,3> hv = one_site_apply(vlo,Vk[i],envr_.back()) * dt;
                for(size_t j=0;j<i+1;++j)
                {
                    T ov = -1.0 * Vk[j].overlap(hv);
                    HmTrips.push_back(make_tuple(j,i,-1.0*ov));
                    hv += Vk[j] * ov;
                }
                T nmhv = hv.norm();
                HmTrips.push_back(make_tuple(i+1,i,nmhv));
                if(i!=MicroIter-1)
                {
                    hv = hv / nmhv;
                    Vk[i+1] = move(hv);
                }
            }
            size_t Nm = Vk.size();
            Tensor<T,2> Hm({Nm,Nm}),eaHm;
            for(auto const & tp : HmTrips)
            {
                Hm({get<0>(tp),get<1>(tp)}) = get<2>(tp);
            }
            eaHm = eig<T>(Hm);
            for(size_t i=0;i<MicroIter;++i)
            {
                eaHm({i,i}) = exp(eaHm({i,i}));
            }
            eaHm = prod<T,2,2,1>(Hm,eaHm);
            eaHm = prod<T,2,2,1>(eaHm,conj<T,2>(Hm),{1},{1});
            Tensor<T,3> res = Vk[0] * eaHm({0,0});
            for(size_t i=1;i<MicroIter;++i)
            {
                res += Vk[i] * eaHm({i,0});
            }
            res =  res * beta;
            return res;
        }
        void two_site_evolve(T const & dt, size_t NumSwps, size_t MicroIter)
        {
            size_t ns = ham_.size();
            T Ene = 0.0, Ovp = 0.0;
            for(size_t swp=0;swp<NumSwps;++swp)
            {
                for(size_t i=0;i<ns-1;++i)
                {
                    Tensor<T,4> s2 = prod<T,3,3,1>(state_[i],state_[i+1],{2},{0});
                    s2 = two_site_gmres(s2,i,two_site_overlap_apply(ovl_.back(),state0_[i],state0_[i+1],ovr_.back()),eta,MicroIter);
                    auto[lef,rig] = svd<T,2,2>(s2,'r',tol_, MaxBond_);
                    state_[i] = move(lef);
                    state_[i+1] = move(rig);
                    if(i!=ns-2)
                    {
                        envl_.push_back(MPO<T>::sweep(envl_.back(),ham_[i],state_[i],state_[i],'r'));
                        envr_.pop_back();
                    }
                }
                for(size_t i=ns-1;i>0;--i)
                {
                    Tensor<T,4> s2 = prod<T,3,3,1>(state_[i-1],state_[i],{2},{0});
                    s2 = two_site_gmres(s2,i-1,two_site_overlap_apply(ovl_.back(),state0_[i-1],state0_[i],ovr_.back()),eta,MicroIter);
                    auto[lef,rig] = svd<T,2,2>(s2,'l',tol_, MaxBond_);
                    state_[i] = move(rig);
                    state_[i-1] = move(lef);
                    if(i!=1)
                    {
                        envr_.push_back(MPO<T>::sweep(envr_.back(),ham_[i],state_[i],state_[i],'l'));
                        envl_.pop_back();
                    }
                }
            }
        }
 
        MPS<T> get_mps() const { return state_; }
        MPO<T> get_mpo() const { return ham_; }

        protected:
        MPO<T> ham_;
        MPS<T> state_;
        vector<MPS<T>> snaps_;
        double tol_;
        size_t MaxBond_;
        vector<Tensor<T,3>> envl_;
        vector<Tensor<T,3>> envr_;
    };
}
