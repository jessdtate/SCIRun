/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2020 Scientific Computing and Imaging Institute,
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


// NOTE: This MatlabIO file is used in different projects as well. Please, do not
// make it depend on other scirun code. This way it is easier to maintain matlabIO
// code among different projects. Thank you.

/*
* FILE: matfilebase.h
* AUTH: Jeroen G Stinstra
* DATE: 16 MAY 2005
*/

#ifndef CORE_MATLABIO_MATFILEBASE_H
#define CORE_MATLABIO_MATFILEBASE_H 1

/*
* matfilebase class includes error handling and definition of
* types. All other matfile classes are derived from this one
* so all enumerations and error handling classes are available
* without spilling everything into the global namespace
* This class should not add any memory overhead on the classes
* derived from it.
*
*/

#include <string>
#include <Core/Exceptions/Exception.h>
#include <Core/Matlab/share.h>

namespace SCIRun
{
  namespace MatlabIO
  {
    class matlabarray;

    class SCISHARE matfilebase
    {
    public:

      // all functions for reading and writing matfiles are based on
      // this class, which defines constants and error exceptions

      // base class for the exceptions
      class matfileerror : public std::exception
      {
      public:
        explicit matfileerror(const std::string& msg = "") : msg_(msg) {}
        const char* what() const NOEXCEPT override
        { return msg_.c_str(); }
      private:
        std::string msg_;
      };

      // Exceptions generated by the different matfile classes
      class could_not_open_file 	: public matfileerror {};
      class io_error				: public matfileerror {};
      class invalid_file_format 	: public matfileerror {};
      class unknown_type			: public matfileerror {};
      class empty_matlabarray     : public matfileerror {};
      class out_of_range			: public matfileerror {};
      class internal_error		: public matfileerror {};
      class invalid_file_access   : public matfileerror {};
      // Added for errors using the compression scheme
      class compression_error		: public matfileerror {};

      // UPDATE:
      // ADDED NEW TAGS FOR MATLAB V7
      enum mitype
      {
        miSAMEASDATA = -1,
        miUNKNOWN = 0,
        miINT8,
        miUINT8,
        miINT16,
        miUINT16,
        miINT32,
        miUINT32,
        miSINGLE,
        miRESV1,
        miDOUBLE,
        miRESV2,
        miRESV3,
        miINT64,
        miUINT64,
        miMATRIX,
        miCOMPRESSED,
        miUTF8,
        miUTF16,
        miUTF32,
        miEND
      };

      // mxtype is the Matlab classification for the Matlab arrays
      // this enum is used internally and is not used at the interface
      // to other classes.
      // Between the mitype and the mxtype there is a certain overlap
      // like a dense matrix is spread out over a number of different
      // classes. To have a nicer interface, a mlclass enum defined
      // only listing the most basic of classes.


      enum mxtype
      {
        mxUNKNOWN = 0,
        mxCELL,
        mxSTRUCT,
        mxOBJECT,
        mxCHAR,
        mxSPARSE,
        mxDOUBLE,
        mxSINGLE,
        mxINT8,
        mxUINT8,
        mxINT16,
        mxUINT16,
        mxINT32,
        mxUINT32
      };

      enum mlclass
      {
        mlUNKNOWN = 0,
        mlCELL,
        mlSTRUCT,
        mlOBJECT,
        mlSTRING,
        mlSPARSE,
        mlDENSE
      };
    };

  }}

#endif
