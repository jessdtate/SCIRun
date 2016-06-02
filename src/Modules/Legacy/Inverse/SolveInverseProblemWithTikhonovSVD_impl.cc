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
#include <Eigen/SVD>

using namespace Eigen;
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
    
    
    JacobiSVD<DenseMatrix> svd( *forwardMatrix, ComputeFullU | ComputeFullV);
    
    
    /*
    const SCIRun::Core::Datatypes::DenseMatrix matrixU_;
    const SCIRun::Core::Datatypes::DenseMatrix matrixS_;
    const SCIRun::Core::Datatypes::DenseMatrix matrixV_;
    */
}


SCIRun::Core::Datatypes::DenseColumnMatrix SolveInverseProblemWithTikhonovSVD_impl::computeInverseSolution( double lambda_sq, bool inverseCalculation )
{
    DenseColumnMatrix solution(3);
    return solution;
}


//
//
//TikhonovSVDAlgorithm::TikhonovSVDAlgorithm(const ColumnMatrix& matrixMeasDatRHS, const DenseMatrix& matrixU, const DenseMatrix& matrixS, const DenseMatrix& matrixV, const DenseMatrix* matrixX, Method method)
//: matrixMeasDatRHS_(matrixMeasDatRHS),
//matrixU_(matrixU),
//matrixS_(matrixS),
//matrixV_(matrixV),
//matrixX_(matrixX)
//{
//    const int M = matrixU.nrows();
//    int N;
//    
//        if (!matrixX_)
//            throw InvalidState("X matrix not specified.");
//        
//        N = matrixX->nrows();
//        const int P = matrixV.nrows();
//        if (M < N)
//        {
//            //throw InvalidState("The forward matrix should be overdetermined.");  //TODO--VERIFY REQUIREMENTS
//        }
//        if (
//            //matrixX->ncols() != N     //TODO--VERIFY REQUIREMENTS
//            //||
//            //matrixS.nrows() != P   //TODO--VERIFY REQUIREMENTS
//            //||
//            //matrixU.ncols() != N   //TODO--VERIFY REQUIREMENTS
//            //||
//            P > N
//            || matrixMeasDatRHS_.nrows() != M)
//        {
//            throw InvalidState("Input matrix dimensions incorrect.");
//        }
//    }
//    
//    int columns = M, solutionRows = N;
//    Uy_ = new ColumnMatrix(matrixU.ncols());
//    inverseMat_ = new DenseMatrix(solutionRows, columns),
//    solution_ = new ColumnMatrix(solutionRows);
//    
//    for (size_type i = 0; i < matrixU_.ncols(); i++)
//        (*Uy_)[i] = inner_product(matrixU_, i, matrixMeasDatRHS_);
//}
//
//TikhonovSVDAlgorithm::~TikhonovSVDAlgorithm() {}
//ColumnMatrixHandle TikhonovSVDAlgorithm::get_solution() const { return solution_; }
//DenseMatrixHandle TikhonovSVDAlgorithm::get_inverse_matrix() const { return inverseMat_; }
//
//////////////////////////////////////////////////////////////////////////////
//// THIS FUNCTION returns the inner product of one column of matrix A
//// and w , B=A(:,i)'*w
/////////////////////////////////////////////////////////////////////////////
//double
//TikhonovSVDAlgorithm::inner_product(const DenseMatrix& A, size_type col_num, const ColumnMatrix& w)
//{
//    int nRows = A.nrows();
//    double B = 0;
//    for (int i = 0; i < nRows; i++)
//        B += A[i][col_num] * w[i];
//    return B;
//}
//
////////////////////////////////////////////////////////////////////////
//// THIS FUNCTION returns regularized solution by tikhonov method
////////////////////////////////////////////////////////////////////////
//
//namespace TikhonovSVDAlgorithmDetail
//{
//    struct IsAlmostZero : public std::unary_function<double, bool>
//    {
//        bool operator()(double d) const
//        {
//            return fabs(d) < 1e-14;
//        }
//    };
//    
//    size_type count_non_zero_entries_in_column(const DenseMatrix& S, size_t column)
//    {
//        return std::count_if(S[column], S[column] + S.nrows(), std::not1(IsAlmostZero()));
//    }
//}
//
//void
//TikhonovSVDAlgorithm::tikhonov_fun(double lambda) const
//{
//    ColumnMatrix& X_reg = *solution_;
//    DenseMatrix& inverseMatrix = *inverseMat_;
//    const ColumnMatrix& Uy = *Uy_;
//    const DenseMatrix& U = matrixU_;
//    const DenseMatrix& S = matrixS_;
//    const DenseMatrix& V = matrixV_;
//    
//    using namespace TikhonovSVDAlgorithmDetail;
//    if (S.ncols() == 1)
//    { // SVD case
//        int rank = count_non_zero_entries_in_column(S, 0);
//        const size_type v_rows = V.nrows();
//        X_reg.zero();
//        for (int i = 0; i < rank; i++)
//        {
//            double filterFactor_i = S[i][0] / (lambda*lambda + S[i][0] * S[i][0]) * Uy[i];
//            for (int j = 0; j < v_rows; j++)
//            {
//                X_reg[j] += filterFactor_i * V[j][i];
//            }
//        }
//        
//        //Finding Regularized Inverse Matrix
//        if (V.ncols() == U.ncols()) //TODO--VERIFY REQUIREMENTS
//        {
//            DenseMatrix Mat_temp(V.nrows(), V.ncols());
//            for (int i = 0; i < rank; i++)
//            {
//                double temp = S[i][0] / (lambda * lambda + S[i][0] * S[i][0]);
//                for (int j = 0; j < V.nrows(); j++)
//                {
//                    Mat_temp[j][i] = temp * V[j][i];
//                }
//            }
//            DenseMatrixHandle Utranspose(U.make_transpose());
//            Mult(inverseMatrix, Mat_temp, *Utranspose);
//        }
//        else
//            inverseMatrix.zero();
//    }
//    else
//    { //GSVD case
//        const DenseMatrix& X = *matrixX_;
//        
//        int rank0 = count_non_zero_entries_in_column(S, 0);
//        int rank1 = count_non_zero_entries_in_column(S, 1);
//        if (rank0 != rank1)
//        {
//            throw InvalidState("singular value vectors do not have same rank");
//        }
//        int rank = rank0;
//        
//        X_reg.zero();
//        
//        ////TODO: make member functions for scalar*row/column, return new column vs in-place
//        
//        for (int i = 0; i < rank; i++)
//        {
//            double filterFactor_i = S[i][0] / (lambda * lambda * S[i][1] * S[i][1] + S[i][0] * S[i][0]) * Uy[i];
//            for (int j = 0; j < X.nrows(); j++)
//            {
//                X_reg[j] += filterFactor_i * X[j][i];
//            }
//        }
//        
//        int minDimension = std::min(U.nrows(), U.ncols());
//        for (int i = rank; i < minDimension; i++)
//        {
//            for (int j = 0; j < X.nrows(); j++)
//            {
//                X_reg[j] += Uy[i] * X[j][i];
//            }
//        }
//		
//        //Finding Regularized Inverse Matrix
//        if (V.ncols() == U.ncols()) //TODO--VERIFY REQUIREMENTS
//        {
//            DenseMatrix Mat_temp = X;
//            for (int i = 0; i < rank; i++)
//            {
//                double temp = S[i][0] / (lambda*lambda*S[i][1] * S[i][1] + S[i][0] * S[i][0]);
//                for (int j = 0; j < X.nrows(); j++)
//                {
//                    Mat_temp[j][i] = temp * X[j][i];
//                }
//            }
//            DenseMatrixHandle Utranspose(U.make_transpose());
//            Mult(inverseMatrix, Mat_temp, *Utranspose);
//        }
//        else
//            inverseMatrix.zero();
//    }










