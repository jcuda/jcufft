/*
 * JCufft - Java bindings for CUFFT, the NVIDIA CUDA FFT library,
 * to be used with JCuda
 *
 * Copyright (c) 2008-2015 Marco Hutter - http://www.jcuda.org
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include "JCufft.hpp"
#include "JCufft_common.hpp"
#include <iostream>
#include <cuda_runtime.h>

jfieldID cufftHandle_plan; // int


/**
 * Initializes JCufft and the CUDA device
 */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *jvm, void *reserved)
{
    JNIEnv *env = NULL;
    if (jvm->GetEnv((void **)&env, JNI_VERSION_1_4))
    {
        return JNI_ERR;
    }
    Logger::log(LOG_TRACE, "Initializing JCufft\n");

    jclass cls = NULL;

    // Initialize the JNIUtils and PointerUtils
    if (initJNIUtils(env) == JNI_ERR) return JNI_ERR;
    if (initPointerUtils(env) == JNI_ERR) return JNI_ERR;

    // Obtain the methodID for cufftHandle#plan
    if (!init(env, cls, "jcuda/jcufft/cufftHandle")) return JNI_ERR;
    if (!init(env, cls, cufftHandle_plan, "plan", "I")) return JNI_ERR;

    return JNI_VERSION_1_4;
}

/**
* Returns the cufftType enum element that corresponds to
* the given int value
*/
cufftType getCufftType(int type)
{
    switch (type)
    {
    case 0x2A: return CUFFT_R2C;
    case 0x2C: return CUFFT_C2R;
    case 0x29: return CUFFT_C2C;
    case 0x6a: return CUFFT_D2Z;
    case 0x6c: return CUFFT_Z2D;
    case 0x69: return CUFFT_Z2Z;
    }
    Logger::log(LOG_ERROR, "Invalid cufftType specified: %d\n", type);
    return CUFFT_C2C;
}


/*
 * Set the log level
 *
 * Class:     jcuda_jcufft_JCufft
 * Method:    setLogLevel
 * Signature: (I)V
 */
JNIEXPORT void JNICALL Java_jcuda_jcufft_JCufft_setLogLevel
  (JNIEnv *env, jclass cla, jint logLevel)
{
    Logger::setLogLevel((LogLevel)logLevel);
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftGetVersionNative
 * Signature: ([I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetVersionNative
  (JNIEnv *env, jclass cls, jintArray version)
{
    if (version == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'version' is null for cufftGetVersion");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetVersion\n");

    int nativeVersion = 0;
    int result = cufftGetVersion(&nativeVersion);
    set(env, version, 0, nativeVersion);
    return result;
}

JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetPropertyNative(JNIEnv *env, jclass cls, jint type, jintArray value)
{
    // Null-checks for non-primitive arguments
    // type is primitive
    if (value == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'value' is null for cufftGetProperty");
        return JCUFFT_INTERNAL_ERROR;
    }

    // Log message
    Logger::log(LOG_TRACE, "Executing cufftGetProperty(type=%d, value=%p)\n",
        type, value);

    // Native variable declarations
    libraryPropertyType type_native;
    int value_native;

    // Obtain native variable values
    type_native = (libraryPropertyType)type;
    // value is write-only

    // Native function call
    cufftResult_t jniResult_native = cufftGetProperty(type_native, &value_native);

    // Write back native variable values
    // type is primitive
    if (!set(env, value, 0, (jint)value_native)) return JCUFFT_INTERNAL_ERROR;

    // Return the result
    jint jniResult = (jint)jniResult_native;
    return jniResult;
}




/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftPlan1dNative
 * Signature: (Ljcuda/jcufft/JCufftHandle;III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftPlan1dNative
  (JNIEnv *env, jclass cla, jobject handle, jint nx, jint type, jint batch)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftPlan1d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Creating 1D plan for %d elements of type %d\n", nx, type);

    cufftHandle plan = env->GetIntField(handle, cufftHandle_plan);
    cufftResult result = cufftPlan1d(&plan, nx, getCufftType(type), batch);
    env->SetIntField(handle, cufftHandle_plan, plan);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftPlan2dNative
 * Signature: (Ljcuda/jcufft/JCufftHandle;III)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftPlan2dNative
  (JNIEnv *env, jclass cla, jobject handle, jint nx, jint ny, jint type)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftPlan2d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Creating 2D plan for (%d, %d) elements of type %d\n", nx, ny, type);

    cufftHandle plan = env->GetIntField(handle, cufftHandle_plan);
    cufftResult result = cufftPlan2d(&plan, nx, ny, getCufftType(type));
    env->SetIntField(handle, cufftHandle_plan, plan);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftPlan3dNative
 * Signature: (Ljcuda/jcufft/JCufftHandle;IIII)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftPlan3dNative
  (JNIEnv *env, jclass cla, jobject handle, jint nx, jint ny, jint nz, jint type)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftPlan3d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Creating 3D plan for (%d, %d, %d) elements of type %d\n", nx, ny, nz, type);

    cufftHandle plan = env->GetIntField(handle, cufftHandle_plan);
    cufftResult result = cufftPlan3d(&plan, nx, ny, nz, getCufftType(type));
    env->SetIntField(handle, cufftHandle_plan, plan);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftPlanManyNative
 * Signature: (Ljcuda/jcufft/cufftHandle;I[I[III[IIIII)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftPlanManyNative
  (JNIEnv *env, jclass cla, jobject handle, jint rank, jintArray n, jintArray inembed, jint istride, jint idist, jintArray onembed, jint ostride, jint odist, jint type, jint batch)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftPlanMany");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cufftPlanMany");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftPlanMany\n");

    cufftHandle plan = env->GetIntField(handle, cufftHandle_plan);
    int *nativeN = getArrayContents(env, n);
    int *nativeInembed = getArrayContents(env, inembed);
    int *nativeOnembed = getArrayContents(env, onembed);

    cufftResult result = cufftPlanMany(&plan, rank, nativeN, nativeInembed, (int)istride, (int)idist, nativeOnembed, (int)ostride, (int)odist, getCufftType(type), (int)batch);

    delete[] nativeN;
    delete[] nativeInembed;
    delete[] nativeOnembed;
    env->SetIntField(handle, cufftHandle_plan, plan);
    return result;

}




/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftMakePlan1dNative
 * Signature: (Ljcuda/jcufft/cufftHandle;III[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftMakePlan1dNative
  (JNIEnv *env, jclass cls, jobject plan, jint nx, jint type, jint batch, jlongArray workSize)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftMakePlan1d");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftMakePlan1d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftMakePlan1d\n");

    cufftHandle nativePlan = env->GetIntField(plan, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftMakePlan1d(nativePlan, (int)nx, getCufftType(type), (int)batch, &nativeWorkSize);

    env->SetIntField(plan, cufftHandle_plan, nativePlan);
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}



/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftMakePlan2dNative
 * Signature: (Ljcuda/jcufft/cufftHandle;III[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftMakePlan2dNative
  (JNIEnv *env, jclass cls, jobject plan, jint nx, jint ny, jint type, jlongArray workSize)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftMakePlan2d");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftMakePlan2d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftMakePlan2d\n");

    cufftHandle nativePlan = env->GetIntField(plan, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftMakePlan2d(nativePlan, (int)nx, (int)ny, getCufftType(type), &nativeWorkSize);

    env->SetIntField(plan, cufftHandle_plan, nativePlan);
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftMakePlan3dNative
 * Signature: (Ljcuda/jcufft/cufftHandle;IIII[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftMakePlan3dNative
  (JNIEnv *env, jclass cls, jobject plan, jint nx, jint ny, jint nz, jint type, jlongArray workSize)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftMakePlan3d");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftMakePlan3d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftMakePlan3d\n");

    cufftHandle nativePlan = env->GetIntField(plan, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftMakePlan3d(nativePlan, (int)nx, (int)ny, (int)nz, getCufftType(type), &nativeWorkSize);

    env->SetIntField(plan, cufftHandle_plan, nativePlan);
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftMakePlanManyNative
 * Signature: (Ljcuda/jcufft/cufftHandle;I[I[III[IIIII[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftMakePlanManyNative
  (JNIEnv *env, jclass cls, jobject plan, jint rank, jintArray n, jintArray inembed, jint istride, jint idist, jintArray onembed, jint ostride, jint odist, jint type, jint batch, jlongArray workSize)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftMakePlanMany");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cufftMakePlanMany");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftMakePlanMany");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftMakePlanMany\n");

    cufftHandle nativePlan = env->GetIntField(plan, cufftHandle_plan);
    int *nativeN = getArrayContents(env, n);
    int *nativeInembed = getArrayContents(env, inembed);
    int *nativeOnembed = getArrayContents(env, onembed);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftMakePlanMany(nativePlan, (int)rank, nativeN, nativeInembed, (int)istride, (int)idist, nativeOnembed, (int)ostride, (int)odist, getCufftType(type), (int)batch, &nativeWorkSize);

    delete[] nativeN;
    delete[] nativeInembed;
    delete[] nativeOnembed;
    env->SetIntField(plan, cufftHandle_plan, nativePlan);
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
* Class:     jcuda_jcufft_JCufft
* Method:    cufftMakePlanManyNative64
* Signature: (Ljcuda/jcufft/cufftHandle;I[J[JJJ[JJJIJ[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftMakePlanManyNative64
(JNIEnv *env, jclass cls, jobject plan, jint rank, jlongArray n, jlongArray inembed, jlong istride, jlong idist, jlongArray onembed, jlong ostride, jlong odist, jint type, jlong batch, jlongArray workSize)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftMakePlanMany64");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cufftMakePlanMany64");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftMakePlanMany64");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftMakePlanMany64\n");

    cufftHandle nativePlan = env->GetIntField(plan, cufftHandle_plan);
    long long *nativeN = getArrayContents(env, n);
    long long *nativeInembed = getArrayContents(env, inembed);
    long long *nativeOnembed = getArrayContents(env, onembed);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftMakePlanMany64(nativePlan, (int)rank, nativeN, nativeInembed, (long long)istride, (long long)idist, nativeOnembed, (long long)ostride, (long long)odist, getCufftType(type), (long long)batch, &nativeWorkSize);

    delete[] nativeN;
    delete[] nativeInembed;
    delete[] nativeOnembed;
    env->SetIntField(plan, cufftHandle_plan, nativePlan);
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}

/*
* Class:     jcuda_jcufft_JCufft
* Method:    cufftGetSizeMany64Native
* Signature: (Ljcuda/jcufft/cufftHandle;I[J[JJJ[JJJIJ[J)I
*/
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetSizeMany64Native
(JNIEnv *env, jclass cls, jobject plan, jint rank, jlongArray n, jlongArray inembed, jlong istride, jlong idist, jlongArray onembed, jlong ostride, jlong odist, jint type, jlong batch, jlongArray workSize)
{
    if (plan == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftGetSizeMany64");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cufftGetSizeMany64");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftGetSizeMany64");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetSizeMany64\n");

    cufftHandle nativePlan = env->GetIntField(plan, cufftHandle_plan);
    long long *nativeN = getArrayContents(env, n);
    long long *nativeInembed = getArrayContents(env, inembed);
    long long *nativeOnembed = getArrayContents(env, onembed);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftGetSizeMany64(nativePlan, (int)rank, nativeN, nativeInembed, (long long)istride, (long long)idist, nativeOnembed, (long long)ostride, (long long)odist, getCufftType(type), (long long)batch, &nativeWorkSize);

    delete[] nativeN;
    delete[] nativeInembed;
    delete[] nativeOnembed;
    env->SetIntField(plan, cufftHandle_plan, nativePlan);
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftEstimate1dNative
 * Signature: (III[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftEstimate1dNative
  (JNIEnv *env, jclass cls, jint nx, jint type, jint batch, jlongArray workSize)
{
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftEstimate1d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftEstimate1d\n");

    size_t nativeWorkSize = 0;
    cufftResult result = cufftEstimate1d((int)nx, getCufftType(type), (int)batch, &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftEstimate2dNative
 * Signature: (III[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftEstimate2dNative
  (JNIEnv *env, jclass cls, jint nx, jint ny, jint type, jlongArray workSize)
{
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftEstimate2d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftEstimate2d\n");

    size_t nativeWorkSize = 0;
    cufftResult result = cufftEstimate2d((int)nx, (int)ny, getCufftType(type), &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftEstimate3dNative
 * Signature: (IIII[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftEstimate3dNative
  (JNIEnv *env, jclass cls, jint nx, jint ny, jint nz, jint type, jlongArray workSize)
{
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftEstimate3d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftEstimate3d\n");

    size_t nativeWorkSize = 0;
    cufftResult result = cufftEstimate3d((int)nx, (int)ny, (int)nz, getCufftType(type), &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftEstimateManyNative
 * Signature: (I[I[III[IIIII[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftEstimateManyNative
  (JNIEnv *env, jclass cls, jint rank, jintArray n, jintArray inembed, jint istride, jint idist, jintArray onembed, jint ostride, jint odist, jint type, jint batch, jlongArray workSize)
{
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cufftEstimateMany");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftEstimateMany");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftEstimateMany\n");

    int *nativeN = getArrayContents(env, n);
    int *nativeInembed = getArrayContents(env, inembed);
    int *nativeOnembed = getArrayContents(env, onembed);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftEstimateMany((int)rank, nativeN, nativeInembed, (int)istride, (int)idist, nativeOnembed, (int)ostride, (int)odist, getCufftType(type), (int)batch, &nativeWorkSize);

    delete[] nativeN;
    delete[] nativeInembed;
    delete[] nativeOnembed;
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftCreateNative
 * Signature: (Ljcuda/jcufft/cufftHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftCreateNative
  (JNIEnv *env, jclass cls, jobject handle)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftCreate");
        return JCUFFT_INTERNAL_ERROR;
    }
    Logger::log(LOG_TRACE, "Executing cufftCreate\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);

    cufftResult result = cufftCreate(&nativeHandle);

    env->SetIntField(handle, cufftHandle_plan, nativeHandle);
    return result;

}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftGetSize1dNative
 * Signature: (Ljcuda/jcufft/cufftHandle;III[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetSize1dNative
  (JNIEnv *env, jclass cls, jobject handle, jint nx, jint type, jint batch, jlongArray workSize)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftGetSize1d");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftGetSize1d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetSize1d\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftGetSize1d(nativeHandle, (int)nx, getCufftType(type), (int)batch, &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftGetSize2dNative
 * Signature: (Ljcuda/jcufft/cufftHandle;III[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetSize2dNative
  (JNIEnv *env, jclass cls, jobject handle, jint nx, jint ny, jint type, jlongArray workSize)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftGetSize2d");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftGetSize2d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetSize2d\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftGetSize2d(nativeHandle, (int)nx, (int)ny, getCufftType(type), &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftGetSize3dNative
 * Signature: (Ljcuda/jcufft/cufftHandle;IIII[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetSize3dNative
  (JNIEnv *env, jclass cls, jobject handle, jint nx, jint ny, jint nz, jint type, jlongArray workSize)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftGetSize3d");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftGetSize3d");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetSize3d\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftGetSize3d(nativeHandle, (int)nx, (int)ny, (int)nz, getCufftType(type), &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;

}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftGetSizeManyNative
 * Signature: (Ljcuda/jcufft/cufftHandle;I[I[III[IIIII[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetSizeManyNative
  (JNIEnv *env, jclass cls, jobject handle, jint rank, jintArray n, jintArray inembed, jint istride, jint idist, jintArray onembed, jint ostride, jint odist, jint type, jint batch, jlongArray workSize)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftGetSizeMany");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (n == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'n' is null for cufftGetSizeMany");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftGetSizeMany");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetSizeMany\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    int *nativeN = getArrayContents(env, n);
    int *nativeInembed = getArrayContents(env, inembed);
    int *nativeOnembed = getArrayContents(env, onembed);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftGetSizeMany(nativeHandle, (int)rank, nativeN, nativeInembed, (int)istride, (int)idist, nativeOnembed, (int)ostride, (int)odist, getCufftType(type), (int)batch, &nativeWorkSize);

    delete[] nativeN;
    delete[] nativeInembed;
    delete[] nativeOnembed;
    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftGetSizeNative
 * Signature: (Ljcuda/jcufft/cufftHandle;[J)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftGetSizeNative
  (JNIEnv *env, jclass cls, jobject handle, jlongArray workSize)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftGetSize");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workSize == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workSize' is null for cufftGetSize");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftGetSize\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    size_t nativeWorkSize = 0;

    cufftResult result = cufftGetSize(nativeHandle, &nativeWorkSize);

    set(env, workSize, 0, (jlong)nativeWorkSize);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftSetWorkAreaNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftSetWorkAreaNative
  (JNIEnv *env, jclass cls, jobject handle, jobject workArea)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftSetWorkArea");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (workArea == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'workArea' is null for cufftSetWorkArea");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftSetWorkArea\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    void *nativeWorkArea = getPointer(env, workArea);

    cufftResult result = cufftSetWorkArea(nativeHandle, nativeWorkArea);

    return result;

}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftSetAutoAllocationNative
 * Signature: (Ljcuda/jcufft/cufftHandle;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftSetAutoAllocationNative
  (JNIEnv *env, jclass cls, jobject handle, jint autoAllocate)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'plan' is null for cufftSetAutoAllocation");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftSetAutoAllocation\n");

    cufftHandle nativeHandle = env->GetIntField(handle, cufftHandle_plan);
    cufftResult result = cufftSetAutoAllocation(nativeHandle, (int)autoAllocate);
    return result;
}





/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftDestroyNative
 * Signature: (Ljcuda/jcufft/JCufftHandle;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftDestroyNative
  (JNIEnv *env, jclass cla, jobject handle)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftDestroy");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Destroying plan\n");

    cufftHandle plan = env->GetIntField(handle, cufftHandle_plan);
    cufftResult result = cufftDestroy(plan);
    return result;
}


//=== Single precision =======================================================

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftExecC2CNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;Ljcuda/Pointer;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftExecC2CNative
  (JNIEnv *env, jclass cla, jobject handle, jobject cIdata, jobject cOdata, jint direction)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftExecC2C");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cIdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cIdata' is null for cufftExecC2C");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cOdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cOdata' is null for cufftExecC2C");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftExecC2C\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    cufftComplex* nativeCIData = (cufftComplex*)getPointer(env, cIdata);
    cufftComplex* nativeCOData = (cufftComplex*)getPointer(env, cOdata);

    cufftResult result = cufftExecC2C(nativePlan, nativeCIData, nativeCOData, direction);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftExecR2CNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftExecR2CNative
  (JNIEnv *env, jclass cla, jobject handle, jobject rIdata, jobject cOdata)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftExecR2C");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (rIdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rIdata' is null for cufftExecR2C");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cOdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cOdata' is null for cufftExecR2C");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftExecR2C\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    float* nativeRIData = (float*)getPointer(env, rIdata);
    cufftComplex* nativeCOData = (cufftComplex*)getPointer(env, cOdata);

    cufftResult result = cufftExecR2C(nativePlan, nativeRIData, nativeCOData);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftExecC2RNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftExecC2RNative
  (JNIEnv *env, jclass cla, jobject handle, jobject cIdata, jobject rOdata)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftExecC2R");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cIdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cIdata' is null for cufftExecC2R");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (rOdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rOdata' is null for cufftExecC2R");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftExecC2R\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    cufftComplex* nativeCIData = (cufftComplex*)getPointer(env, cIdata);
    float* nativeROData = (float*)getPointer(env, rOdata);

    cufftResult result = cufftExecC2R(nativePlan, nativeCIData, nativeROData);
    return result;
}



//=== Double precision =======================================================



/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftExecZ2ZNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;Ljcuda/Pointer;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftExecZ2ZNative
  (JNIEnv *env, jclass cla, jobject handle, jobject cIdata, jobject cOdata, jint direction)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftExecZ2Z");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cIdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cIdata' is null for cufftExecZ2Z");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cOdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cOdata' is null for cufftExecZ2Z");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftExecZ2Z\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    cufftDoubleComplex* nativeCIData = (cufftDoubleComplex*)getPointer(env, cIdata);
    cufftDoubleComplex* nativeCOData = (cufftDoubleComplex*)getPointer(env, cOdata);

    cufftResult result = cufftExecZ2Z(nativePlan, nativeCIData, nativeCOData, direction);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftExecD2ZNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftExecD2ZNative
  (JNIEnv *env, jclass cla, jobject handle, jobject rIdata, jobject cOdata)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftExecD2Z");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (rIdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rIdata' is null for cufftExecD2Z");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cOdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cOdata' is null for cufftExecD2Z");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftExecD2Z\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    double* nativeRIData = (double*)getPointer(env, rIdata);
    cufftDoubleComplex* nativeCOData = (cufftDoubleComplex*)getPointer(env, cOdata);

    cufftResult result = cufftExecD2Z(nativePlan, nativeRIData, nativeCOData);
    return result;
}

/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftExecZ2DNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/Pointer;Ljcuda/Pointer;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftExecZ2DNative
  (JNIEnv *env, jclass cla, jobject handle, jobject cIdata, jobject rOdata)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftExecZ2D");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (cIdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'cIdata' is null for cufftExecZ2D");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (rOdata == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'rOdata' is null for cufftExecZ2D");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftExecZ2D\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    cufftDoubleComplex* nativeCIData = (cufftDoubleComplex*)getPointer(env, cIdata);
    double* nativeROData = (double*)getPointer(env, rOdata);

    cufftResult result = cufftExecZ2D(nativePlan, nativeCIData, nativeROData);
    return result;
}


/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftSetStreamNative
 * Signature: (Ljcuda/jcufft/cufftHandle;Ljcuda/runtime/cudaStream_t;)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftSetStreamNative
  (JNIEnv *env, jclass cla, jobject handle, jobject stream)
{
    if (handle == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'handle' is null for cufftSetStream");
        return JCUFFT_INTERNAL_ERROR;
    }
    if (stream == NULL)
    {
        ThrowByName(env, "java/lang/NullPointerException", "Parameter 'stream' is null for cufftSetStream");
        return JCUFFT_INTERNAL_ERROR;
    }

    Logger::log(LOG_TRACE, "Executing cufftSetStream\n");

    cufftHandle nativePlan = env->GetIntField(handle, cufftHandle_plan);
    cudaStream_t nativeStream = NULL;
    nativeStream = (cudaStream_t)getNativePointerValue(env, stream);

    cufftResult result = cufftSetStream(nativePlan, nativeStream);
    return result;
}



/*
 * Class:     jcuda_jcufft_JCufft
 * Method:    cufftSetCompatibilityModeNative
 * Signature: (Ljcuda/jcufft/cufftHandle;I)I
 */
JNIEXPORT jint JNICALL Java_jcuda_jcufft_JCufft_cufftSetCompatibilityModeNative
  (JNIEnv *env, jclass cla, jobject plan, jint mode)
{
	ThrowByName(env, "java/lang/UnsupportedOperationException", "Function cufftSetCompatibilityMode was removed in CUDA version 9.1.");
	return JCUFFT_INTERNAL_ERROR;
}
