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


#include <Modules/Legacy/Inverse/SolveInverseProblemWithTikhonovSVD.h>
#include <Modules/Legacy/Inverse/SolveInverseProblemWithTikhonovSVD_impl.h>
#include <Modules/Legacy/Inverse/TikhonovImplAbstractBase.h>
#include <Core/Datatypes/MatrixTypeConversions.h>
#include <Core/Datatypes/DenseColumnMatrix.h>
#include <Core/Datatypes/Legacy/Field/Field.h>


using namespace SCIRun;
using namespace SCIRun::Core::Datatypes;
using namespace SCIRun::Modules::Inverse;
using namespace SCIRun::Dataflow::Networks;
using namespace SCIRun::Core::Algorithms;


/////////////////////////////////////////
// MODULE EXECUTION
/////////////////////////////////////////
void SolveInverseProblemWithTikhonovSVD::execute()
{
        auto forward_matrix_h = getRequiredInput(ForwardMatrix);
        auto hMatrixMeasDat = getRequiredInput(MeasuredPotentials);

        
        const bool computeRegularizedInverse = oport_connected(InverseSolution);
        
        if (needToExecute())
        {
            
            using namespace BioPSE;
            auto state = get_state();

            
            auto denseForward = castMatrix::toDense(forward_matrix_h);
            auto measuredDense = convertMatrix::toDense(hMatrixMeasDat);
    
            
            SolveInverseProblemWithTikhonovSVD_impl algo(   denseForward,
                                                            measuredDense,
                                                            computeRegularizedInverse,
                                                            this);
            
            TikhonovImplAbstractBase::Input::lcurveGuiUpdate update = boost::bind(&SolveInverseProblemWithTikhonovSVD::update_lcurve_gui, this, _1, _2, _3);
            
            TikhonovImplAbstractBase::Input input(
                                                    state->getValue(RegularizationMethod).toString(),
                                                    state->getValue(LambdaFromDirectEntry).toDouble(),
                                                    state->getValue(LambdaSliderValue).toDouble(),
                                                    state->getValue(LambdaNum).toInt(),
                                                    state->getValue(LambdaMin).toDouble(),
                                                    state->getValue(LambdaMax).toDouble(),
                                                    update);
            
            
            algo.run(input);
            
            if (computeRegularizedInverse)
            {
                sendOutput(RegInverse, algo.get_inverse_matrix());
            }
            
            sendOutput(InverseSolution, algo.get_inverse_solution());
            
            sendOutput(RegularizationParameter, algo.get_regularization_parameter());
        }
        
}

