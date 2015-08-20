package jcuda.jcufft;




import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Basic test of the bindings of the JCufft class 
 */
public class JCufftBasicBindingTest
{
    public static void main(String[] args)
    {
        JCufftBasicBindingTest test = new JCufftBasicBindingTest();
        test.testJCufft();
    }

    @Test
    public void testJCufft()
    {
        assertTrue(BasicBindingTest.testBinding(JCufft.class));
    }
    
}
