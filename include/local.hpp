#pragma once
#include "util.hpp"
template<typename ArrType>
ArrType to_matrix(char op, size_t dim)
{
    ArrType res({1,dim,dim,1});
    size_t lev = 0;
    switch (op)
    {
    case 'I':
        for(lev=0;lev<dim;++lev)
        {
            res({0,lev,lev,0}) = 1.0;
        }
        break;
    case 'Q':
        for(lev=0;lev<dim-1;++lev)
        {
            res({0,lev,lev+1,0}) = std::sqrt(0.5*lev+0.5);
            res({0,lev+1,lev,0}) = std::sqrt(0.5*lev+0.5);
        }
        break;
    case '+':
        for(lev=0;lev<dim-1;++lev)
        {
            res({0,lev+1,lev,0}) = 1.0*std::sqrt(1.0*lev+1.0);
        }
        break;
    case '-':
        for(lev=0;lev<dim-1;++lev)
        {
            res({0,lev,lev+1,0}) = 1.0*std::sqrt(1.0*lev+1.0);
        }
        break;
    case '^': // Q2
        for(lev=0;lev<dim-2;++lev)
        {
            res({0,lev,lev+2,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
            res({0,lev+2,lev,0}) = std::sqrt(0.5*lev+0.5)*std::sqrt(0.5*lev+1.0);
        }
        for(lev=0;lev<dim;++lev)
        {
            res({0,lev,lev,0}) = 0.5*(2*lev+1.0);
        }
        break;
    default:
        break;
    }
    return res;
}

