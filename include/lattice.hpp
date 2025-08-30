#pragma once
#include "nil_mpo.h"
using namespace Nil;
struct Lattice
{
    size_t nsites = 0;
    vector<size_t> nphysdims;
    vector<vector<char>> types;
    vector<double> rvals;
    vector<double> ivals;

    Lattice() = default;
    ~Lattice() = default;
    Lattice(const char * file_name)
    {
        size_t KTerm = 0;
        ifstream ifs(fdump);
        if(!ifs.is_open()) { cout << "Fail to open dump!" << endl; exit(1);}
        string lines,segs;
        getline(ifs,lines);
        stringstream ss_lines(lines);
        while(getline(ss_lines,segs,' '))
        {
            nphysdims.push_back(stoul(segs));
        }
        nsites = nphysdims.size();
        while(!ifs.eof())
        {
            string lines,segs;
            getline(ifs,lines);
            stringstream ss_lines(lines);
            vector<string> slices;
            while(getline(ss_lines,segs,' '))
            {
                slices.push_back(segs);
            }
            size_t num_element = slices.size();
            size_t num_ops = (num_element - 2) / 2;
            vector<char> cur_type(nsites,'I');
            for(size_t loc=0;loc<num_ops;++loc)
            {
                cur_type[stoul(slices[loc])] = slices[loc+num_ops][0];
            }
            rvals.push_back(stod(slices[2*num_ops]));
            ivals.push_back(stod(slices[2*num_ops]));
            types.push_back(cur_type);
            if(ifs.eof()) {break;}
        }
        ifs.close();
    }
    MPO<double> gen_total_para()
    {
        size_t nTerm = rvals.size(), nth;
        #pragma omp parallel
        nth = omp_get_num_threads();

        vector<MPO<double>> hams(nth,nsites);
        #pragma omp parallel for
        for(size_t i=0;i<nTerm;++i)
        {
            MPO<double> tmpi(nsites);
            for(size_t j=0;j<nsites;++j)
            {
                tmpi[j] = to_matrix<Tensor<double,4>>(types[i][j],nphysdims[j]);
            }
            tmpi *= rvals[i];
            size_t ThreadID = omp_get_thread_num();
            if(hams[ThreadID][0].shape()[1]!=tmpi[0].shape()[1]) { hams[ThreadID] = tmpi; }
            else { hams[ThreadID] += tmpi; }
            hams[ThreadID].canon();
        }
        for(size_t i=1;i<nth;++i)
        {
            hams[0] += hams[i];
            hams[0].canon();
        }
        return hams[0];
    }
    MPO<MKL_Complex16> gen_ctotal_para()
    {
        size_t nTerm = rvals.size(), nth;
        #pragma omp parallel
        nth = omp_get_num_threads();

        vector<MPO<MKL_Complex16>> hams(nth,nsites);
        #pragma omp parallel for
        for(size_t i=0;i<nTerm;++i)
        {
            MPO<MKL_Complex16> tmpi(nsites);
            for(size_t j=0;j<nsites;++j)
            {
                tmpi[j] = to_matrix<Tensor<MKL_Complex16,4>>(types[i][j],nphysdims[j]);
            }
            MKL_Complex16 val(rvals[i],ivals[i]);
            tmpi[0] = tmpi * val; 
            size_t ThreadID = omp_get_thread_num();
            if(hams[ThreadID][0].shape()[1]!=tmpi[0].shape()[1]) { hams[ThreadID] = tmpi; }
            else { hams[ThreadID] += tmpi; }
            hams[ThreadID].canon();
        }
        for(size_t i=1;i<nth;++i)
        {
            hams[0] += hams[i];
            hams[0].canon();
        }
        return hams[0];
    }
};
