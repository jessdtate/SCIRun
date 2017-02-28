/*
For more information, please see: http://software.sci.utah.edu

The MIT License

Copyright (c) 2015 Scientific Computing and Imaging Institute,
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

//    File       : SolveInverseProblemWithTikhonov.cc
//    Author     : Moritz Dannhauer, Ayla Khan, Dan White
//    Date       : November 02th, 2012 (last update)

#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>

#include <Core/Algorithms/Legacy/Inverse/TikhonovImplAbstractBase.h>

#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/DenseColumnMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/MatrixTypeConversions.h>

#include <Core/Algorithms/Base/AlgorithmPreconditions.h>

#include <Core/Logging/LoggerInterface.h>
#include <Core/Utils/Exception.h>


using namespace SCIRun;
using namespace SCIRun::Core;
using namespace SCIRun::Core::Datatypes;
using namespace SCIRun::Core::Logging;
using namespace SCIRun::Core::Algorithms;
using namespace SCIRun::Core::Algorithms::Inverse;

    TikhonovImplAbstractBase::TikhonovImplAbstractBase(const DenseMatrixHandle& forwardMatrix,
                                                 const DenseMatrixHandle& measuredData,
                                                 AlgorithmChoice regularizationChoice,
                                                 AlgorithmSolutionSubcase regularizationSolutionSubcase,
                                                 AlgorithmResidualSubcase regularizationResidualSubcase,
                                                 const DenseMatrixHandle sourceWeighting,
                                                 const DenseMatrixHandle sensorWeighting,
                                                 bool computeRegularizedInverse,
                                                 LegacyLoggerInterface* pr)
    : forwardMatrix_(forwardMatrix),
    measuredData_(measuredData),
    sourceWeighting_(sourceWeighting),
    sensorWeighting_(sensorWeighting),
    regularizationChoice_(regularizationChoice),
    regularizationSolutionSubcase_(regularizationSolutionSubcase),
    regularizationResidualSubcase_(regularizationResidualSubcase),
    lambda_(0),
    computeRegularizedInverse_(computeRegularizedInverse),
    pr_(pr)
    {
        // check input sizes
        checkInputMatrixSizes();

    }


    MatrixHandle TikhonovImplAbstractBase::get_inverse_solution() const
    {
        return inverseSolution_;
    }

    MatrixHandle TikhonovImplAbstractBase::get_inverse_matrix() const
    {
        return inverseMatrix_;
    }

    DenseColumnMatrixHandle TikhonovImplAbstractBase::get_regularization_parameter() const
    {
        return regularizationParameter_;
    }


////// CHECK IF INPUT MATRICES HAVE THE CORRECT SIZE
    bool TikhonovImplAbstractBase::checkInputMatrixSizes()
    {

        const int M = forwardMatrix_->nrows();
        const int N = forwardMatrix_->ncols();

        // check that rows of fwd matrix equal number of measurements
        if ( M != measuredData_->nrows() )
        {
            const std::string errorMessage("Input matrix dimensions must agree.");
            if (pr_)
            {
                pr_->error(errorMessage);
            }
            else
            {
                std::cerr << errorMessage << std::endl;
            }
            BOOST_THROW_EXCEPTION(DimensionMismatch() << DimensionMismatchInfo(errorMessage));
        }

        // check that number of time samples is 1. @JCOLLFONT to change for a more general case later (should add a for loop)
        if (1 != measuredData_->ncols())
        {
            const std::string errorMessage("Measured data must be a vector");
            if (pr_)
            {
                pr_->error(errorMessage);
            }
            else
            {
                std::cerr << errorMessage << std::endl;
            }
            BOOST_THROW_EXCEPTION(DimensionMismatch() << DimensionMismatchInfo(errorMessage));
        }

        // check source regularization matrix sizes
        if (sourceWeighting_)
        {
            if( regularizationSolutionSubcase_==solution_constrained )
            {
                // check that the matrix is of appropriate size (equal number of rows as columns in fwd matrix)
                if ( N != sourceWeighting_->ncols() )
                {
                    const std::string errorMessage("Solution Regularization Matrix must have the same number of rows as columns in the Forward Matrix !");
                    if (pr_)
                    {
                        pr_->error(errorMessage);
                    }
                    else
                    {
                        std::cerr << errorMessage << std::endl;
                    }
                    BOOST_THROW_EXCEPTION(DimensionMismatch() << DimensionMismatchInfo(errorMessage));
                }
            }
            // otherwise, if the source regularization is provided as the squared version (RR^T)
            else if ( regularizationSolutionSubcase_==solution_constrained_squared )
            {
                // check that the matrix is of appropriate size and squared (equal number of rows as columns in fwd matrix)
                if ( ( N != sourceWeighting_->nrows() ) || ( N != sourceWeighting_->ncols() ) )
                {
                    const std::string errorMessage("The squared solution Regularization Matrix must have the same number of rows and columns and must be equal to the number of columns in the Forward Matrix !");
                    if (pr_)
                    {
                        pr_->error(errorMessage);
                    }
                    else
                    {
                        std::cerr << errorMessage << std::endl;
                    }
                    BOOST_THROW_EXCEPTION(DimensionMismatch() << DimensionMismatchInfo(errorMessage));
                }
            }
        }

        // check measurement regularization matrix sizes
        if (sensorWeighting_)
        {
            if (regularizationResidualSubcase_ == residual_constrained)
            {
                // check that the matrix is of appropriate size (equal number of rows as rows in fwd matrix)
                if(M != sensorWeighting_->ncols())
                {
                    const std::string errorMessage("Data Residual Weighting Matrix must have the same number of rows as the Forward Matrix !");
                    if (pr_)
                    {
                        pr_->error(errorMessage);
                    }
                    else
                    {
                        std::cerr << errorMessage << std::endl;
                    }
                    BOOST_THROW_EXCEPTION(DimensionMismatch() << DimensionMismatchInfo(errorMessage));
                }
            }
            // otherwise if the source covariance matrix is provided in squared form
            else if  ( regularizationResidualSubcase_ == residual_constrained_squared )
            {
                // check that the matrix is of appropriate size and squared (equal number of rows as rows in fwd matrix)
                if( (M != sensorWeighting_->nrows()) || (M != sensorWeighting_->ncols()) )
                {
                    const std::string errorMessage("Squared data Residual Weighting Matrix must have the same number of rows and columns as number of rows in the Forward Matrix !");
                    if (pr_)
                    {
                        pr_->error(errorMessage);
                    }
                    else
                    {
                        std::cerr << errorMessage << std::endl;
                    }
                    BOOST_THROW_EXCEPTION(DimensionMismatch() << DimensionMismatchInfo(errorMessage));
                }

            }
        }

        return true;

    }


/////////////////////////
/////////  run()
    void TikhonovImplAbstractBase::run(const TikhonovImplAbstractBase::Input& input)
    {

        const int M = forwardMatrix_->nrows();

        // Alocate variables
        DenseColumnMatrix solution(M);
        double lambda_sq=0;


        //Get Regularization parameter(s) : Lambda
        if ((input.regMethod_ == "single") || (input.regMethod_ == "slider"))
        {
            if (input.regMethod_ == "single")
            {
                // Use single fixed lambda value, entered in UI
                lambda_ = input.lambdaFromTextEntry_;
            }
            else if (input.regMethod_ == "slider")
            {
                // Use single fixed lambda value, select via slider
                lambda_ = input.lambdaSlider_;
            }
        }
        else if (input.regMethod_ == "lcurve")
        {
            lambda_ = computeLcurve( input );
        }


        lambda_sq = lambda_ * lambda_;

        // compute inverse solution
        solution = computeInverseSolution( lambda_sq, computeRegularizedInverse_);

        // set final result
        inverseSolution_.reset(new DenseMatrix(solution));

        // output regularization parameter
        DenseColumnMatrix tempLambda(1);
        tempLambda[0] = lambda_;

        regularizationParameter_. reset( new DenseColumnMatrix(tempLambda) );


    }
//////// fi  run()
///////////////////////////


///////////////////////////
/////// compute L-curve
    double TikhonovImplAbstractBase::computeLcurve( const TikhonovImplAbstractBase::Input& input )
    {

        // define the step size of the lambda vector to be computed  (distance between min and max divided by number of desired lambdas in log scale)
        const int nLambda = input.lambdaCount_;
        const double lam_step = pow(10.0, log10(input.lambdaMax_ / input.lambdaMin_) / (nLambda-1));
        double lambda;

        const int sizeSolution = forwardMatrix_->ncols();
        double lambda_sq;

        // prealocate vector of lambdas and eta and rho
        std::vector<double> lambdaArray(nLambda, 0.0);
        std::vector<double> rho(nLambda, 0.0);
        std::vector<double> eta(nLambda, 0.0);

        DenseColumnMatrix CAx, Rx;
        DenseColumnMatrix solution(sizeSolution);

        lambdaArray[0] = input.lambdaMin_;

        // initialize counter
        int lambda_index = 0;

        // for all lambdas
        for (int j = 0; j < nLambda; j++)
        {
            if (j)
            {
                lambdaArray[j] = lambdaArray[j-1] * lam_step;
            }

            // set current lambda
            lambda_sq = lambdaArray[j] * lambdaArray[j];

            // COMPUTE INVERSE SOLUTION  // Todo: @JCOLLFONT function needs to be defined
            solution = computeInverseSolution( lambda_sq, false);


            // if using source regularization matrix, apply it to compute Rx (for the eta computations)
            if (sourceWeighting_)
            {
                if (solution.nrows() == sourceWeighting_->ncols()) // check that regularization matrix and solution match sizes
                    Rx = *sourceWeighting_ * solution;
                else
                {
                    const std::string errorMessage(" Solution weighting matrix unexpectedly does not fit to compute the weighted solution norm. ");
                    if (pr_)
                        pr_->error(errorMessage);
                    else
                        std::cerr << errorMessage << std::endl;
                }
            }
            else
                Rx = solution;


            auto Ax = *forwardMatrix_  * solution;
            auto residualSolution = Ax - *measuredData_;

            // if using source regularization matrix, apply it to compute Rx (for the eta computations)
            if (sensorWeighting_)
                CAx = (*sensorWeighting_) * residualSolution;
            else
                CAx = residualSolution;


            // compute rho and eta
            rho[j]=0; eta[j]=0;
            for (int k = 0; k < CAx.nrows(); k++)
            {
                double T = CAx(k);
                rho[j] += T*T; //norm of the data fit term
            }

            for (int k = 0; k < Rx.nrows(); k++)
            {
                double T = Rx[k];
                eta[j] += T*T; //norm of the model term
            }

            // eta and rho needed to plot the Lcurve and determine the L corner
            rho[j] = sqrt(rho[j]);
            eta[j] = sqrt(eta[j]);


        }

        // update L-curve
        boost::shared_ptr<TikhonovAlgorithm::LCurveInput> lcurveInput(new TikhonovAlgorithm::LCurveInput(rho, eta, lambdaArray, nLambda));
        lcurveInput_handle_ = lcurveInput;

        // Find corner in L-curve
        lambda = FindCorner(*lcurveInput_handle_, lambda_index);

        // update GUI
        if (input.updateLCurveGui_)
            input.updateLCurveGui_(lambda, *lcurveInput_handle_, lambda_index);

        // return lambda
        return lambda;

    }
//////// fi compute L-curve
/////////////////////////////


///// Find Corner, find the maximal curvature which corresponds to the L-curve corner
    double TikhonovImplAbstractBase::FindCorner(const TikhonovAlgorithm::LCurveInput& input, int& lambda_index)
    {
        const std::vector<double>& rho = input.rho_;
        const std::vector<double>& eta = input.eta_;
        const std::vector<double>& lambdaArray = input.lambdaArray_;
        int nLambda = input.nLambda_;

        std::vector<double> deta(nLambda);
        std::vector<double> ddeta(nLambda);
        std::vector<double> drho(nLambda);
        std::vector<double> ddrho(nLambda);
        std::vector<double> lrho(nLambda);
        std::vector<double> leta(nLambda);
        DenseColumnMatrix kapa(nLambda);

        double maxKapa = -1.0e10;
        for (int i = 0; i < nLambda; i++)
        {
            lrho[i] = std::log10(rho[i]);
            leta[i] = std::log10(eta[i]);
            if(i>0)
            {
                deta[i] = (leta[i]-leta[i-1]) / (lambdaArray[i]-lambdaArray[i-1]); // compute first derivative
                drho[i] = (lrho[i]-lrho[i-1]) / (lambdaArray[i]-lambdaArray[i-1]);
            }
            if(i>1)
            {
                ddeta[i] = (deta[i]-deta[i-1]) / (lambdaArray[i]-lambdaArray[i-1]); // compute second derivative from first
                ddrho[i] = (drho[i]-drho[i-1]) / (lambdaArray[i]-lambdaArray[i-1]);
            }
        }
        drho[0] = drho[1];
        deta[0] = deta[1];
        ddrho[0] = ddrho[2];
        ddrho[1] = ddrho[2];
        ddeta[0] = ddeta[2];
        ddeta[1] = ddeta[2];

        lambda_index = 0;
        for (int i = 0; i < nLambda; i++)
        {
            kapa[i] = std::abs((drho[i] * ddeta[i] - ddrho[i] * deta[i]) /  //compute curvature
                               std::sqrt(std::pow((deta[i]*deta[i]+drho[i]*drho[i]), 3.0)));
            if (kapa[i] > maxKapa) // find max curvature
            {
                maxKapa = kapa[i];
                lambda_index = i;
            }
        }

        return lambdaArray[lambda_index];
    }

///// Search for closest Lambda to given lambda
    double TikhonovImplAbstractBase::LambdaLookup(const TikhonovAlgorithm::LCurveInput& input, double lambda, int& lambda_index, const double epsilon)
    {
        const std::vector<double>& lambdaArray = input.lambdaArray_;
        int nLambda = input.nLambda_;

        for (int i = 0; i < nLambda-1; ++i)
        {
            if (i > 0 && (lambda < lambdaArray[i-1] || lambda > lambdaArray[i+1])) continue;

            double lambda_step_midpoint = std::abs(lambdaArray[i+1] - lambdaArray[i])/2;

            if (std::abs(lambda - lambdaArray[i]) <= epsilon)  // TODO: is this a reasonable comparison???
            {
                lambda_index = i;
                return lambdaArray[lambda_index];
            }

            if (std::abs(lambda - lambdaArray[i]) < lambda_step_midpoint)
            {
                lambda_index = i;
                return lambdaArray[lambda_index];
            }

            if (std::abs(lambda - lambdaArray[i+1]) < lambda_step_midpoint)
            {
                lambda_index = i+1;
                return lambdaArray[lambda_index];
            }
        }
        return -1;
    }


/////// UI interaction functionality ////////

////////// update L-curve graph
    void TikhonovImplAbstractBase::update_graph(const TikhonovImplAbstractBase::Input& input, double lambda, int lambda_index, const double epsilon)
    {
        if (lcurveInput_handle_ && input.updateLCurveGui_)
        {
            lambda = LambdaLookup(*lcurveInput_handle_, lambda, lambda_index, epsilon);
            if (lambda >= 0)
            {
                input.updateLCurveGui_(lambda, *lcurveInput_handle_, lambda_index);
            }
        }
    }


//// Set eta, rho and num lambda
    TikhonovAlgorithm::LCurveInput::LCurveInput(const std::vector<double>& rho, const std::vector<double>& eta, const std::vector<double>& lambdaArray, int nLambda)
    : rho_(rho), eta_(eta), lambdaArray_(lambdaArray), nLambda_(nLambda)
    {}

//// Set input from UI
    TikhonovImplAbstractBase::Input::Input(const std::string& regMethod, double lambdaFromTextEntry, double lambdaSlider, int lambdaCount, double lambdaMin, double lambdaMax,
                                        lcurveGuiUpdate updateLCurveGui)
    : regMethod_(regMethod), lambdaFromTextEntry_(lambdaFromTextEntry), lambdaSlider_(lambdaSlider), lambdaCount_(lambdaCount), lambdaMin_(lambdaMin), lambdaMax_(lambdaMax),
    updateLCurveGui_(updateLCurveGui)
    {}
