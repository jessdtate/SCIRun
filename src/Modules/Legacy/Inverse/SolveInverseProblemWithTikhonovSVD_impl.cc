/*
 For more information, please see: http://software.sci.utah.edu
 
 The MIT License
 
 Copyright (c) 2009 Scientific Computing and Imaging Institute,
 University of Utah.
 
 
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 DEALINGS IN THE SOFTWARE.
 */

//    File       : SolveInverseProblemWithTikhonovSVD.cc
//    Author     : Yesim Serinagaoglu & Alireza Ghodrati
//    Date       : 07 Aug. 2001
//    Last update: Dec 2011


// SCIRUN lybraries
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/MatrixTypeConversions.h>
#include <Core/Algorithms/Base/AlgorithmPreconditions.h>
#include <Core/Logging/LoggerInterface.h>
#include <Core/Utils/Exception.h>

// Tikhonov inverse libraries
#include <Modules/Legacy/Inverse/TikhonovImplAbstractBase.h>
#include <Modules/Legacy/Inverse/SolveInverseProblemWithTikhonovSVD.h>
#include <Modules/Legacy/Inverse/SolveInverseProblemWithTikhonovSVD_impl.h>

// EIGEN LIBRARY
//#include <Eigen/Eigen>
#include <Eigen/SVD>


using namespace BioPSE;
using namespace SCIRun;
using namespace SCIRun::Core::Datatypes;
using namespace SCIRun::Modules::Inverse;
using namespace SCIRun::Dataflow::Networks;
using namespace SCIRun::Core::Algorithms;



/////// prealocate Matrices for inverse compuation
///     This function precalcualtes the SVD of the forward matrix and prepares singular vectors and values for posterior computations
void SolveInverseProblemWithTikhonovSVD_impl::preAlocateInverseMatrices()
{
    
    // Compute the SVD of the forward matrix
        Eigen::JacobiSVD<Eigen::MatrixXf> SVDdecomposition( castMatrix::toDense(forwardMatrix_), Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // Set matrix U, S and V
        matrixU_ = SVDdecomposition.matrixU();
        matrixV_ = SVDdecomposition.matrixV();
        matrixS_ = SVDdecomposition.singularValues();
    
    // Compute the projection of data y on the left singular vectors
        Uy = matrixU_.transpose() * y;
    
}

//////////////////////////////////////////////////////////////////////
// THIS FUNCTION returns regularized solution by tikhonov method
//////////////////////////////////////////////////////////////////////
SCIRun::Core::Datatypes::DenseColumnMatrix SolveInverseProblemWithTikhonovSVD_impl::computeInverseSolution( double lambda_sq, bool inverseCalculation )
{
    DenseColumnMatrix solution(3);
    
    const int M = matrixU.nrows();
    const int N = matrixV.nrows();
    
    
    // Check rank of fwd matrix
    int rank = count_non_zero_entries_in_column(S, 0);
    
    // Compute inverse solution
    for (int rr=o; rr<rank ; rr++)
    {
        double filterFactor_i = matrixS_[rr] / ( lambda_sq + matrixS_[rr] * matrixS_[rr] ) * Uy[i];
        
        solution += filterFactor_i * matrixV_.column(rr);
    }
    
    
    // Compute inverse matrix if required
    if (inverseCalculation)
    {
        DenseMatrix tempInverseMatrix(Eigen::zero(N,M));
        
        for (int rr=o; rr<rank ; rr++)
        {
            double filterFactor_i = matrixS_[rr] / ( lambda_sq + matrixS_[rr] * matrixS_[rr] ) * Uy[i];
            inverseMatrix_ += filterFactor_i * ( matrixV_.column(rr) *  matrixU_.column(rr).transpose() );
        }
        inverseMatrix_.reset( tempInverseMatrix );
    }
    
    // output solution
    return solution;
}



//////////////////////////////////////////////////////////////////
///////// This functions evaluate if an entry is close to 0
//////////////////////////////////////////////////////////////////
namespace TikhonovSVDAlgorithmDetail
{
    struct IsAlmostZero : public std::unary_function<double, bool>
    {
        bool operator()(double d) const
        {
            return fabs(d) < 1e-14;
        }
    };
        
    size_type count_non_zero_entries_in_column(const DenseMatrix& S, size_t column)
    {
        return std::count_if(S[column], S[column] + S.nrows(), std::not1(IsAlmostZero()));
    }
}
////////////////



