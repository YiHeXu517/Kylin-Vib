#pragma once
#include "nil_tensor.hpp"
#include "nil_hash.hpp"
namespace Nil
{
    template<typename T, size_t N>
    Tensor<T,N> conj(Tensor<T,N> const & r)
    {
        Tensor<T,N> res(r);
        if constexpr (is_same<T,MKL_Complex16>::value)
        {
            #pragma omp parallel for
            for(size_t i=0;i<r.data.size();++i)
            {
                res.data[i] = conj(r.data[i]);
            }
        }
        return res;
    }
    // transpose
    template<typename T, size_t N>
    Tensor<T,N> transpose(Tensor<T,N> const & r, array<size_t,N> const & ax)
    {
        char UnTransposed = 'y';
        for(size_t i=0;i<N-1;++i)
        {
            if(ax[i+1] != ax[i]+1)
            {
                UnTransposed = 'n';
                break;
            }
        }
        if(UnTransposed=='y')
        {
            Tensor<T,N> res(r);
            return res;
        }
        array<size_t,N> rsp;
        for(size_t i=0;i<N;++i)
        {
            rsp[i] = r.shape[ax[i]];
        } 
        Tensor<T,N> res(rsp);
        #pragma omp parallel for
        for(size_t i=0;i<r.data.size();++i)
        {
            array<size_t,N> idx;
            for(size_t j=0;j<N;++j)
            {
                idx[j] = i / r.dist[ax[j]] % r.shape[ax[j]];
            }
            res(idx) = r.data[i];
        }
        return res;
    }
    template<typename T, size_t N>
    Tensor<T,N> transpose(Tensor<T,N> const & r, initializer_list<size_t> ax)
    {
        array<size_t,N> axr;
        copy(ax.begin(),ax.end(),axr.begin());
        return transpose(r,axr);
    }
    // stack
    template<typename T, size_t N>
    Tensor<T,N> stack(Tensor<T,N> const & r1, Tensor<T,N> const & r2, initializer_list<size_t> ax)
    {
        array<size_t,N> rsp(r1.shape);
        for(auto it = ax.begin(); it!= ax.end(); ++it)
        {
            rsp[*it] += r2.shape[*it];
        }
        Tensor<T,N> res(rsp);
        for(size_t i=0;i<r1.data.size();++i)
        {
            array<size_t,N> idx;
            for(size_t j=0;j<N;++j)
            {
                idx[j] = i / r1.dist[j] % r1.shape[j];
            }
            res(idx) = r1.data[i];
        }
        for(size_t i=0;i<r2.data.size();++i)
        {
            array<size_t,N> idx;
            for(size_t j=0;j<N;++j)
            {
                idx[j] = i / r2.dist[j] % r2.shape[j];
            }
            for(auto it = ax.begin(); it!= ax.end(); ++it)
            {
                idx[*it] += r1.shape[*it];
            }
            res(idx) = r2.data[i];
        }
        return res;
    }
    // svd return (u,s,vt)
    template<typename T, size_t R1, size_t R2>
    tuple<Tensor<T,R1+1>,Tensor<T,R2+1>> svd(Tensor<T,R1+R2> & m, char l2r = 'r', double tol = 1e-14, size_t maxdim = 1000,
    char ReS = 'n')
    {
        size_t nrow = accumulate(m.shape.begin(),m.shape.begin()+R1,1,multiplies<size_t>());
        size_t ncol = m.data.size() / nrow;
        size_t ldu = min(nrow,ncol);
        Tensor<T,R1+R2> mc(m);
        Tensor<T,2> u({nrow,ldu}),vt({ldu,ncol});
        Tensor<double,1> s({ldu}),sp({ldu-1});
        array<size_t,R1+1> lsp; copy(m.shape.begin(),m.shape.begin()+R1,lsp.begin());
        array<size_t,R2+1> rsp; copy(m.shape.begin()+R1,m.shape.end(),rsp.begin()+1);
        if constexpr (is_same<T,double>::value)
        {
            size_t ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.data.data(), ncol, s.data.data(), u.data.data(), ldu, vt.data.data(), ncol, sp.data.data());
            if(ifsv!=0) { cout << "Problems in SVD! " << R1 << " : " << R2 << endl; exit(1);}
        }
        else if constexpr (is_same<T,MKL_Complex16>::value)
        {
            size_t ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.data.data(), ncol, s.data.data(), u.data.data(), ldu, vt.data.data(), ncol, sp.data.data());
            if(ifsv!=0) { cout << "Problems in SVD! " << R1 << " : " << R2 << endl; exit(1);}
        }
        size_t nstate = 1; double nrms = 0;
        for(size_t i=0;i<ldu;++i)
        {
            nrms += s.data[i];
            if(s.data[i]<=tol)
            {
                    break;
            }
            nstate = i + 1;
        }
        nstate = min(maxdim,nstate);
        lsp[R1] = nstate; rsp[0] = nstate;
        Tensor<T,R1+1> lef(lsp); Tensor<T,R2+1> rig(rsp);
        if constexpr (is_same<T,double>::value)
        {
            if(l2r=='r')
            {
                for(size_t i=0;i<nstate;++i)
                {
                    cblas_dcopy(nrow, u.data.data()+i, ldu, lef.data.data()+i, nstate);
                    cblas_daxpy(ncol, s.data[i], vt.data.data()+i*ncol, 1, rig.data.data()+i*ncol, 1);
                }
                if(ReS=='r') { lef *= sqrt(nrms); rig *= (1/sqrt(nrms));}
            }
            else
            {
                for(size_t i=0;i<nstate;++i)
                {
                    cblas_daxpy(nrow, s.data[i],    u.data.data()+i, ldu, lef.data.data()+i, nstate);
                    cblas_dcopy(ncol, vt.data.data()+i*ncol, 1, rig.data.data()+i*ncol, 1);
                }
                if(ReS=='r') { lef *= (1/sqrt(nrms)); rig *= sqrt(nrms); }
            }
        }
        else if constexpr (is_same<T,MKL_Complex16>::value)
        {
            if(l2r=='r')
            {
                for(size_t i=0;i<nstate;++i)
                {
                    T sc(s.data[i]);
                    cblas_zcopy(nrow, u.data.data()+i, ldu, lef.data.data()+i, nstate);
                    cblas_zaxpy(ncol, &sc, vt.data.data()+i*ncol, 1, rig.data.data()+i*ncol, 1);
                }
                if(ReS=='r') { lef *= sqrt(nrms); rig *= (1/sqrt(nrms));}
            }
            else
            {
                for(size_t i=0;i<nstate;++i)
                {
                    T sc(s.data[i]);
                    cblas_zaxpy(nrow, &sc, u.data.data()+i, ldu, lef.data.data()+i, nstate);
                    cblas_zcopy(ncol, vt.data.data()+i*ncol, 1, rig.data.data()+i*ncol, 1);
                }
                if(ReS=='r') { lef *= (1/sqrt(nrms)); rig *= sqrt(nrms); }
            }
        }
        return make_tuple(lef,rig);
    }
    // eigensolver
    template<typename T>
    Tensor<T,2> eig(Tensor<T,2> & Hm, char rep = 'r')
    {
        size_t Ns = Hm.shape[0];
        int ifeg;
        Tensor<double,1> ega({Ns});
        Tensor<T,2> egr({Ns,Ns});
        if constexpr (is_same<T,double>::value)
        {
            ifeg = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.data.data(), Ns, ega.data.data());
            if(ifeg!=0) { cout << "Problems in eig! " << Ns << endl; exit(1);}
        }
        else if constexpr (is_same<T,MKL_Complex16>::value)
        {
            ifeg = LAPACKE_zheevd(LAPACK_ROW_MAJOR, 'V', 'U', Ns, Hm.data.data(), Ns, ega.data.data());
            if(ifeg!=0) { cout << "Problems in eig! " << Ns << endl; exit(1);}
        }
        for(size_t i=0;i<Ns;++i)
        {
            egr({i,i}) = ega({i});
        }
        return egr;                                                                                                                                                                                                                                                                                                                 
    }
    // generate transpose shape
    template<typename T, size_t N, size_t rc>
    Tensor<T,N> prod_trans( Tensor<T,N> const & r, 
    initializer_list<size_t> ax, char l2r)
    {
        array<size_t,N> AxTrans;
        if(l2r=='l')
        {
            size_t anchor = 0;
            for(size_t i=0;i<N;++i)
            {
                if(find(ax.begin(),ax.end(),i)==ax.end())
                {
                    AxTrans[anchor] = i;
                    anchor++;
                }
            }
            copy(ax.begin(),ax.end(),AxTrans.begin()+N-rc);
        }
        else
        {
            size_t anchor = 0;
            for(size_t i=0;i<N;++i)
            {
                if(find(ax.begin(),ax.end(),i)==ax.end())
                {
                    AxTrans[anchor+rc] = i;
                    anchor++;
                }
            }
            copy(ax.begin(),ax.end(),AxTrans.begin());
        }
        return transpose<T,N>(r,AxTrans);
    }
    // combine resulting shapes
    template<typename T, size_t R1, size_t R2, size_t rc>
    array<T,R1+R2-2*rc> prod_shape(array<T,R1> const & sp1,
    array<T,R2> const & sp2, initializer_list<size_t> ax1, 
    initializer_list<size_t> ax2)
    {
        array<T,R1+R2-2*rc> res;
        size_t anchor = 0;
        for(size_t i=0;i<R1;++i)
        {
            if(find(ax1.begin(),ax1.end(),i)==ax1.end())
            {
                res[anchor] = sp1[i];
                anchor++;
            }
        }
        for(size_t i=0;i<R2;++i)
        {
            if(find(ax2.begin(),ax2.end(),i)==ax2.end())
            {
                res[anchor] = sp2[i];
                anchor++;
            }
        }
        return res;
    }
    // calculate the needed parameters for gemm_
    template<typename T, size_t R1, size_t R2, size_t rc>
    array<size_t,3> prod_params(Tensor<T,R1> const & r1t, Tensor<T,R2> const & r2t)
    {
        array<size_t,3> rck;
        rck[0] = accumulate(r1t.shape.begin(),r1t.shape.begin()+R1-rc,1,multiplies<size_t>());
        rck[1] = accumulate(r2t.shape.begin()+rc,r2t.shape.end(),1,multiplies<size_t>());
        rck[2] = r1t.data.size() / rck[0];
        if(rck[2] != r2t.data.size()/rck[1] )
        {
            cout << "r1@r2 shapes mismatch!" << endl;
            r1t.print(1e9); r2t.print(1e9);
            exit(1);
        }
        return rck;
    }
    // product main implementation
    template<typename T, size_t R1, size_t R2, size_t rc>
    Tensor<T,R1+R2-2*rc> prod(Tensor<T,R1> const & r1, Tensor<T,R2> const & r2,
    initializer_list<size_t> ax1, initializer_list<size_t> ax2)
    {
        Tensor<T,R1> r1t = prod_trans<T,R1,rc>(r1,ax1,'l');
        Tensor<T,R2> r2t = prod_trans<T,R2,rc>(r2,ax2,'r');
        array<size_t,R1+R2-2*rc> rsp = prod_shape<size_t,R1,R2,rc>(r1.shape,r2.shape,ax1,ax2);
        Tensor<T,R1+R2-2*rc> res(rsp);
        array<size_t,3> rcks = prod_params<T,R1,R2,rc>(r1t,r2t);
        T ones(1.0),zeros(0.0);
        if constexpr (is_same<T,double>::value)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], ones, 
            r1t.data.data(), rcks[2], r2t.data.data(), rcks[1], zeros, res.data.data(), rcks[1]);
        }
        else if constexpr (is_same<T,MKL_Complex16>::value)
        {
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], &ones, 
            r1t.data.data(), rcks[2], r2t.data.data(), rcks[1], &zeros, res.data.data(), rcks[1]);
        }
        return res;
    }
    // matrix product for labelled array
    template<typename T>
    LabArr<T,2> mat_prod(LabArr<T,2> const & c1, LabArr<T,2> const & c2)
    {
        LabArr<T,2> res({c1.shape[0],c2.shape[1]});
        res.labs = c1.labs;
        res.labs.insert(res.labs.end(), c2.labs.begin(), c2.labs.end());
        T ones(1.0),zeros(0.0);
        array<size_t,3> rcks = {c1.shape[0],c2.shape[1],c1.shape[1]};
        if constexpr (is_same<T,double>::value)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], ones, 
            c1.data.data(), rcks[2], c2.data.data(), rcks[1], zeros, res.data.data(), rcks[1]);
        }
        else if constexpr (is_same<T,MKL_Complex16>::value)
        {
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rcks[0], rcks[1], rcks[2], &ones, 
            c1.data.data(), rcks[2], c2.data.data(), rcks[1], &zeros, res.data.data(), rcks[1]);
        }
        return res;
    }
    template<typename T>
    Tensor<T,2> svd_pinv(Tensor<T,2> & m)
    {
        size_t nrow = m.shape[0];
        size_t ncol = m.shape[1];
        size_t ldu = min(nrow,ncol);
        int ifsv;
        Tensor<T,2> u({nrow,ldu}),vt({ldu,ncol}),diags({ldu,ldu});
        Tensor<double,1> s({ldu}),sp({ldu-1});
        if constexpr (is_same<T,double>::value)
        {
        ifsv = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.data.data(), ncol,
                        s.data.data(), u.data.data(), ldu, vt.data.data(), ncol, sp.data.data());
        }
        else if constexpr (is_same<T,MKL_Complex16>::value)
        {
        ifsv = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, 'S', 'S', nrow, ncol, m.data.data(), ncol,
                        s.data.data(), u.data.data(), ldu, vt.data.data(), ncol, sp.data.data());
        }
        if(ifsv!=0) { cout << "Problems in SVD! " << endl; exit(1);}
        for(size_t i=0;i<ldu;++i)
        {
            diags({i,i}) = 1.0 / s.data[i];
        }
        Tensor<T,2> res = prod<T,2,2,1>(vt,diags,{0},{0});
        res = prod<T,2,2,1>(res,u,{1},{1});
        return res;
    }
}
