package IRLcode;

import burlap.behavior.functionapproximation.dense.DenseStateFeatures;
import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.singleagent.Episode;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateReachability;
import burlap.behavior.singleagent.auxiliary.valuefunctionvis.ValueFunctionVisualizerGUI;
import burlap.behavior.singleagent.learnfromdemo.RewardValueProjection;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRL;
import burlap.behavior.singleagent.learnfromdemo.mlirl.MLIRLRequest;
import burlap.behavior.singleagent.learnfromdemo.mlirl.commonrfs.LinearStateDifferentiableRF;
import burlap.behavior.singleagent.learnfromdemo.mlirl.differentiableplanners.DifferentiableSparseSampling;
import burlap.behavior.valuefunction.QProvider;
import burlap.debugtools.RandomFactory;
import burlap.domain.singleagent.gridworld.GridWorldDomain;
import burlap.domain.singleagent.gridworld.GridWorldVisualizer;
import burlap.domain.singleagent.gridworld.state.GridAgent;
import burlap.domain.singleagent.gridworld.state.GridLocation;
import burlap.domain.singleagent.gridworld.state.GridWorldState;
import burlap.mdp.auxiliary.StateGenerator;
import burlap.mdp.core.oo.OODomain;
import burlap.mdp.core.oo.propositional.GroundedProp;
import burlap.mdp.core.oo.propositional.PropositionalFunction;
import burlap.mdp.core.oo.state.OOState;
import burlap.mdp.core.state.State;
import burlap.mdp.singleagent.environment.SimulatedEnvironment;
import burlap.mdp.singleagent.oo.OOSADomain;
import burlap.shell.visual.VisualExplorer;
import burlap.statehashing.simple.SimpleHashableStateFactory;
import burlap.visualizer.Visualizer;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.lang.math.RandomUtils;
import org.apache.commons.math3.linear.RealVectorFormat;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.lang.System;

import static com.sun.tools.classfile.AccessFlags.Kind.Field;
import static corejava.Format.printf;
import static jdk.nashorn.internal.objects.Global.println;
import static org.apache.commons.lang.math.RandomUtils.nextDouble;
import static org.apache.commons.math3.linear.MatrixUtils.createRealIdentityMatrix;
import static org.apache.commons.math3.linear.MatrixUtils.inverse;

public class IRLsampler {

    public static int Nloc;
    public static int Nrf;

    public static int maxX;
    public static int maxY;

    public static String pathToEpisodes = "demos";


    public IRLsampler(int Nloc, int Nrf, int maxX, int maxY) {

        this.Nloc = Nloc;
        this.Nrf  = Nrf;        // total number of rewards to be sampled
        this.maxX = maxX;
        this.maxY = maxY;
    }


    /**
    / ---------------------------------------------------------------------------------------------------------------- /
    /                                                 Sampling Rewards                                                 /
    / ---------------------------------------------------------------------------------------------------------------- /
    **/


    public RealVector[] sampleRewards(RealMatrix[] TransP, RealMatrix Ppi) {

        System.out.format("\n ----- Sampling rewards: ----- \n\n");
        RealVector[] sampled_rewards = new RealVector[Nrf];

        RealVector baseR = new ArrayRealVector(new double[maxX * maxY]); // initial point in the space to search: zero vector

        double thd = 1E-5;

        int rCount = 0; int lCount = 0;
        while (rCount < Nrf) {

            RealVector dirU  = new ArrayRealVector(generateRandomDirecVector());   // randomly generated direction vector

            double edgetPos = getPosEdgeT(baseR, dirU, TransP, Ppi, thd);
            double edgetNeg = getNegEdgeT(baseR, dirU, TransP, Ppi, thd);
            baseR           = getNewBaseR(edgetNeg, edgetPos, baseR, dirU);

         System.out.format("\nPositive edge t: %.20f; Negative edge t: %.20f\n\n", edgetPos, edgetNeg);

            if (lCount % 10 == 0) { sampled_rewards[rCount] = baseR; rCount += 1; }
            lCount += 1;
        }
        return sampled_rewards;
    }


    public double getPosEdgeT(RealVector baseR, RealVector dirU, RealMatrix[] TransP, RealMatrix Ppi, double thd) {

        System.out.format("exp | positive\n\n");
        double hightPos = expSearch(1E-4, baseR, dirU, TransP, Ppi);
        double lowtPos  = 0.5 * hightPos;                                                                           // System.out.format("\nlowt: %.20f, hight: %.20f\n\n", lowtPos, hightPos);

        System.out.format("\nbin | positive\n\n");
        return binSearch(thd, lowtPos, hightPos, baseR, dirU, TransP, Ppi);
    }


    public double getNegEdgeT(RealVector baseR, RealVector dirU, RealMatrix[] TransP, RealMatrix Ppi, double thd) {

        System.out.format("\nexp | negative\n\n");
        double hightNeg = expSearch(-1E-4, baseR, dirU, TransP, Ppi);
        double lowtNeg  = 0.5 * hightNeg;                                                                           // System.out.format("\nlowt: %.20f, hight: %.20f\n\n", lowtPos, hightPos);

        System.out.format("\nbin | negative\n\n");
        return  binSearch(thd, lowtNeg, hightNeg, baseR, dirU, TransP, Ppi);
    }


    private double expSearch(double t, RealVector baseR, RealVector dirU,
                                     RealMatrix[] TransP, RealMatrix Ppi) {

        if (!isValid(baseR.add(dirU.mapMultiply(t)), TransP, Ppi) ) { return t; }

        t = t + t;
        return expSearch(t, baseR, dirU, TransP, Ppi);
    }

    private double binSearch(double thd, double lowt, double hight,
                                     RealVector baseR, RealVector dirU, RealMatrix[] TransP, RealMatrix Ppi) {

        if (java.lang.Math.abs(hight-lowt) < thd) { return lowt; }

        double midt = 0.5 * (hight + lowt);

        if (isValid(baseR.add(dirU.mapMultiply(midt)), TransP, Ppi)) {
            return binSearch(thd, midt, hight, baseR, dirU, TransP, Ppi);
        } else {
            return binSearch(thd, lowt, midt, baseR, dirU, TransP, Ppi);
        }
    }

    public boolean isValid(RealVector newR, RealMatrix[] TransP, RealMatrix Ppi) {
        if ( !withinRange(newR) ) { return false; }

        int        N   = maxX*maxY;
        double     miu = 0.1;

        boolean isPositive = true;

        checkLoop:
        for (int s = 0; s < N; s++) {
            for (int a = 0; a < 4; a++) {

                System.out.format("Checking on s %d, a %d\n", s, a);

                RealVector Ppi_s    = Ppi.getRowVector(s);
                RealVector TransP_s = TransP[a].getRowVector(s);

                /** ------------------------------------- testing ------------------------------------- **/
//                System.out.format("\n--- Current R: \n");
//                for (int i = 0; i < maxX*maxY; i++) { System.out.format("%.20f ", R.getEntry(i)); }
//                System.out.format("\n\n");
//
//                System.out.format("I - miu Ppi: \n");
//                for (int i = 0; i < maxX*maxY; i++) {
//                    for (int j = 0; j < maxX*maxY; j++) {
//                        System.out.format("%.20f ", I.subtract(Ppi.scalarMultiply(miu) ).getEntry(i,j));
//                    }
//                    System.out.format("\n");
//                }
//                System.out.format("\n");
//
//                System.out.format("inverse: \n");
//                for (int i = 0; i < maxX*maxY; i++) {
//                    for (int j = 0; j < maxX*maxY; j++) {
//                        System.out.format("%.30f ", inverse( I.subtract(Ppi.scalarMultiply(miu) )).getEntry(i,j));
//                    }
//                    System.out.format("\n");
//                }
//                System.out.format("\n");
//
//                System.out.format("Ppi-P: \n");
//                for (int i = 0; i < maxX*maxY; i++) { System.out.format("%.20f ", Ppi_s.subtract(TransP_s).getEntry(i)); }
//                System.out.format("\n\n");
//
//                System.out.format("Ppi-P * matrix: \n");
//                for (int i = 0; i < maxX*maxY; i++) {
//                    System.out.format("%.20f ", inverse( I.subtract(Ppi.scalarMultiply(miu) )).preMultiply( Ppi_s.subtract(TransP_s) ).getEntry(i));
//                }
//                System.out.format("\n\n");
//
//                System.out.format("Ppi-P * matrix * R: \n");
//                System.out.format("%.20f \n\n", inverse( I.subtract(Ppi.scalarMultiply(miu) )).preMultiply( Ppi_s.subtract(TransP_s) ).dotProduct( R ));
                /** ------------------------------------- testing ------------------------------------- **/

                isPositive = positiveResult(Ppi, Ppi_s, TransP_s, newR, miu, N);
                if (!isPositive) { break checkLoop; }
            }
        }
        return isPositive;
    }


    public boolean positiveResult(RealMatrix Ppi, RealVector Ppi_s, RealVector TransP_s, RealVector R, double miu, int dimension) {

        RealMatrix I      = createRealIdentityMatrix(dimension);
        RealVector U      = inverse( I.subtract(Ppi.scalarMultiply(miu)) ).preMultiply( Ppi_s.subtract(TransP_s) );
        double     result = U.dotProduct( R );

        System.out.format( "Validity checking result: %.20f\n\n", result );

        return result >= 0;
    }


    public boolean withinRange(RealVector newR) {

        for (int i = 0; i < newR.getDimension(); i++) {
            if (newR.getEntry(i) > 1.0 | newR.getEntry(i) < -1.0) { return false; }
        }
        return true;
    }

    // -- helper -- //
    private RealVector getNewBaseR(double edget_neg, double edget_pos,
                                             RealVector baseR, RealVector dirU) {
        double randt = edget_neg + (edget_pos - edget_neg) * nextDouble();
        return baseR.add(dirU.mapMultiply(randt));
    }

    // -- helper -- //
    public double[] generateRandomDirecVector() {

        double radius = 1.0;
        int    len    = maxX * maxY;

        double[] randU                 = new double[len];
        double[] randSphereCoordinates = new double[len];

        for (int i = 1; i < len-1; i++) {randSphereCoordinates[i] = randDouble(0.0, Math.PI, true, true);}
        randSphereCoordinates[len-1] = randDouble(0.0, 2*Math.PI, true, false);

        for (int j = 1; j < len; j++) { randU[j-1] = radius * cosineProduct(randSphereCoordinates,j) * Math.cos(randSphereCoordinates[j]); }
        randU[len-1] = radius * cosineProduct(randSphereCoordinates, len);

        return randU;
    }


    public double cosineProduct(double[] randSphereCoordinates, int endIndex) {

        double product = 1;
        for (int i = 1; i < endIndex; i++) {
            product = product * Math.sin(randSphereCoordinates[i]);
        }
        return product;
    }

    // -- helper -- //
    public double randDouble(double lowerBound, double higherBound, boolean includeLow, boolean includeHigh) {
        double newDouble = lowerBound + (higherBound - lowerBound) * nextDouble();

        if (!includeHigh && (newDouble == higherBound)) { return randDouble(lowerBound, higherBound, includeLow, includeHigh); }
        if (!includeLow  && (newDouble == lowerBound))  { return randDouble(lowerBound, higherBound, includeLow, includeHigh); }

        return newDouble;
    }


    /** ------------------------------------------------- MAIN ----------------------------------------------------- **/


    public void main(String[] args) throws IOException {

//        String pathToEpisodes = "demos/4demo0.episode";
//        episodeParser EP = new episodeParser(Nloc, maxX, maxY, pathToEpisodes);

//        RealVector[] Rs = sampleRewards(EP.TransP, EP.Ppi);


    }
}





//            System.out.format("\n  Current base R: \n");
//            for (int i = 0; i < maxX*maxY; i++) { System.out.format("%.20f ", baseR.getEntry(i)); }
//            System.out.format("\n\n");
//
//            System.out.format("\n  Current direc U: \n");
//            for (int i = 0; i < maxX*maxY; i++) { System.out.format("%.20f ", dirU.getEntry(i)); }
//            System.out.format("\n\n");


// Exponential Search (for space boundary)
//            RealVector newR = new ArrayRealVector(maxX * maxY);

//            t = 10;
//            newR = baseR.add(dirU.mapMultiply(t));
//
//            while (isValid(newR, TransP, Pi, Ppi)) {
//                System.out.format("valid t: %.20f\n", t);
//                t = t + t;
//                newR = baseR.add(dirU.mapMultiply(t));
//            }

//            for (t = 1E-4; isValid(newR, TransP, Pi, Ppi); t = t+t) { newR = baseR.add(dirU.mapMultiply(t)); }                                                                                                           // TODO: init t set to 0.001


//            while (hight - lowt >= thd) {
//                double midt = 0.5 * (hight + lowt);
//                if (isValid(baseR.add(dirU.mapMultiply(midt)), TransP, Pi, Ppi)) {lowt = midt;} else {hight = midt;}
////                System.out.format("lowt: %.20f, hight: %.20f\n", lowt, hight);
//            }
//            double edget1 = lowt;
//            System.out.format("\nHigher edge t: %.20f\n", edget1);



//            // Exponential Search (for space boundary)
//            newR = new ArrayRealVector(maxX * maxY);        // TODO: init t set to 0.001
//
//            t = -10;
//            newR = baseR.add(dirU.mapMultiply(t));
//
//            while (isValid(newR, TransP, Pi, Ppi)) {
//                t = t + t;
//                newR = baseR.add(dirU.mapMultiply(t));
//            }

//            for (t = -0.000000000001; isValid(newR, TransP, Pi, Ppi); t = t+t) { newR = baseR.add(dirU.mapMultiply(t)); }



//            // Binary Search (for edge t)
//            lowt = t; hight = 0.5 * t;
//            System.out.format("\nlowt: %.20f, hight: %.20f\n\n", lowt, hight);
//
//            while (hight - lowt >= thd) {
//                double midt = 0.5 * (hight + lowt);
//                if (isValid(baseR.add(dirU.mapMultiply(midt)), TransP, Pi, Ppi)) {hight = midt;} else {lowt = midt;}
////                System.out.format("lowt: %.20f, hight: %.20f\n", lowt, hight);
//            }
//            double edget2 = hight;