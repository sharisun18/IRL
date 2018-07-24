package IRLcode;

/** Episode Parser
 *
 *  This object is responsible for getting necessary information from episode files.(For now, from one episode file)
 *
 *  To initialize a parser, the constructor requires:
 *
 *      1) Nloc (number of location types)
 *      2) maxX & maxY (size of the map)
 *      3) path_to_episode_file
 *
 *  At the time of initialization, the object will contain following data structures:
 *
 *      1) TransP - Transition probability matrix:
 *
 *              Type:     RealMatrix[4]
 *              Content: [action_number][start_state][end_state]
 *
 *                  The transition prob matrix is actually independent of the environment and policy of the observation.
 *              It only depends on the size of the map. The corresponding transition prob of an action is default to be
 *              1, but when the state is an "edge state", the transition prob of an action leading to the wall will lead
 *              to the state itself.
 *
 *      2) ENV - Environment:
 *
 *              Type:     GridLocation[maxX * maxY].
 *
 *              Each item is a list containing:
 *                        1) rowN
 *                        2) colN
 *                        3) locType
 *                        4) locName (just a nominal name that is different for each location, not having any meaning)
 *      3) Pi - Policy:
 *
 *              Type:    RealVector
 *              Content: Each group-of-four is a list (n_p, e_p, s_p, w_p),
 *                       denoting probability of taking each direction in the leading state.
 *
 *      4) Ppi - Transition Probability under current policy:
 *
 *              Type:    RealMatrix, of size (maxX*maxY) * (maxX*maxY)
 *              Content:
 *
 *  Other functions of a parser:
 *
 *      - getStartPosition()
 */

import burlap.domain.singleagent.gridworld.state.GridLocation;
import org.apache.commons.math3.linear.*;

import java.util.*;
import java.lang.System;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static corejava.Format.printf;

public class episodeParser {

    private static int Nloc;
    private static int maxX;
    private static int maxY;
    private static int N = maxX * maxY;

    private static String pathToEpisodes;
    private static String[] episode;
    private static int epiFileLength;

    public RealMatrix[]   TransP;
    public GridLocation[] ENV;
    public RealVector     Pi;
    public RealMatrix     Ppi;


    // Constructors
    public episodeParser(int Nloc, int maxX, int maxY) {
        this.Nloc = Nloc;
        this.maxX = maxX;
        this.maxY = maxY;
    }


    public episodeParser(int Nloc, int maxX, int maxY, String pathToEpisodes) throws IOException {

        this.Nloc = Nloc;
        this.maxX = maxX;
        this.maxY = maxY;

        this.pathToEpisodes = pathToEpisodes;
        this.episode = readEpisode(this.pathToEpisodes);
        this.epiFileLength = this.episode.length;

        this.TransP = getTransitionProbs();
        this.ENV    = parseEnvironment();
        this.Pi     = getPolicyFromTrajectory();
        this.Ppi    = getPpi(Pi, TransP);
    }

    // Read episodes from episodes folder.                                                      TODO: for now, only read one episode file (given path to the file)
    private String[] readEpisode(String pathToEpisodes) throws IOException {

        Scanner epiFile          = new Scanner(new File(pathToEpisodes)).useDelimiter("\n");
        List<String> episodeList = new ArrayList<String>();

        while (epiFile.hasNext()) { episodeList.add(epiFile.nextLine()); }

        return episodeList.toArray(new String[0]);
    }


    /* -------------------------------------------------------------------------------------------------------------- */
    /*                                                Major Functions                                                 */
    /* -------------------------------------------------------------------------------------------------------------- */


    /** ----------------------------------------------- Environment ------------------------------------------------ **/

    /* parseEnvironment
        Given one episode file, parse environment from episode,
        location type, name, and axis are stored as one state in *locations* (which is an array)
     */
    public static GridLocation[] parseEnvironment() {

        System.out.format("Parsing environment\n\n");

        // Store loc type at all domain locations / set of GridLocations
        GridLocation[] locations = new GridLocation[maxX * maxY];

        for (int i = 0; i < epiFileLength; i++) {

            // Find the part denoting state distribution (environment), read contents to locations
            if (episode[i].contentEquals("stateSequence:")) {
                i += 4;
                while ( episode[i].charAt(0) == ' ' ) {
                        String[] chunks = episode[i].split(": ");

                    String locName = chunks[1].substring(0,chunks[1].length()-6);
                    int    locType = Integer.parseInt(chunks[2].substring(0,1));
                    int    rowN    = Integer.parseInt(chunks[3].substring(0,1));
                    int    colN    = Integer.parseInt(chunks[4].substring(0,1));


//                    System.out.format("%s %d %d %d\n", locName, locType, rowN, colN);                       // Print out current location info


                    locations[maxX*rowN+colN] = new GridLocation(rowN, colN, locType, locName);
                    i++;
                }
                break;
            }
        }
        return locations;
    }


    /** -------------------------------------------- Trajectory & Policy ------------------------------------------- **/

    /* parseTrajectory
        It returns the observed trajectory as an array of state-action tri-set: [x, y, action]
        0: north; 1: east; 2: south; 3: west
    */
    public static int[][] parseTrajectory() {

        System.out.format("Parsing trajectory\n\n");

        // Get number of actions and initialize T
        int startRow = 0; int endRow = 0;
        for (int i = 0; i < epiFileLength; i++) {
            if (episode[i].contentEquals("actionSequence:")) { startRow = i; }
            if (episode[i].contentEquals("stateSequence:" )) { endRow   = i; }
        }
        int Naction = endRow-startRow-2;
        int[][] T = new int[Naction][3];

        // Walk through trajectory
        int[] prevLoc = getStartPosition();
        int[] curLoc  = new int[2];

        int r = 0; int j = endRow + 2;
        while ( r < Naction ) {
            if (episode[j].substring(0,1).contentEquals("-")) {

                curLoc[0]    = Integer.parseInt(episode[j+1].split(": ")[3].substring(0,1));
                curLoc[1]    = Integer.parseInt(episode[j+1].split(": ")[4].substring(0,1));
                int direction = getDirection(prevLoc, curLoc);

//                System.out.format("%d, %d, %d\n", prevLoc[0], prevLoc[1], direction);

                T[r][0] = prevLoc[0]; T[r][1] = prevLoc[1]; T[r][2] = direction;
                prevLoc[0] = curLoc[0]; prevLoc[1] = curLoc[1];
                r++;
            }
            j++;
        }
        return T;
    }

    /* getStartPosition
        Get initialization of the agent before episode starts
    */
    public static int[] getStartPosition() {

        int[] initLoc = new int[2];

        for (int i = 0; i < epiFileLength; i++) {
            if (episode[i].contentEquals("stateSequence:")) {
                i += 2;
                String[] chunks = episode[i].split(": ");
                initLoc[0] = Integer.parseInt(chunks[3].substring(0,1));
                initLoc[1] = Integer.parseInt(chunks[4].substring(0,1));
                break;
            }
        }
        return initLoc;
    }

    // -- helper --
    private static int getDirection(int[] prevLoc, int[] curLoc) {

        int d_x = curLoc[0] - prevLoc[0];
        int d_y = curLoc[1] - prevLoc[1];
        if (d_y == 1) { return 0; } else if (d_x == 1) { return 1; } else if (d_y == -1) { return 2; } else if (d_x == -1) { return 3; }

        return -1;
    }

    /* getPolicyFromTrajectory
        Get trajectory from episode, and construct an arbitrary policy that contains the trajectory.
        The policy will be stored as a long one-dimensional array, with every four denoting P(s, a_i) for four directions.
     */
    public static RealVector getPolicyFromTrajectory() {

        System.out.format("Generating policy\n\n");

        double[] Pi = new double[maxX * maxY * 4];

        // Parse trajectory from episode
        int[][]  T  = parseTrajectory();

        // Put trajectory into policy
        for (int i = 0; i < T.length; i++) {
            int x = T[i][0]; int y = T[i][1]; int direc = T[i][2];
            Pi [ x*20 + y*4 + direc ] = 1;
        }

        // Set other unaccessed state actions
        for (int j = 0; j < Pi.length; j += 4) {
            if (isUnaccessedState(Pi, j))
                 { Pi[j] = Pi[j+1] = Pi[j+2] = Pi[j+3] = 0.25; }
            else { standardizeAccessedStates(Pi, j); }
        }

        // Print out policy
        for (int i = 0; i < Pi.length; i += 4) {
            System.out.format("%d: %f %f %f %f\n", i/4, Pi[i], Pi[i+1], Pi[i+2], Pi[i+3]);
            if ((i/4+1) % 5 == 0) { System.out.format("\n"); }
        }

        return new ArrayRealVector(Pi);
    }

    // -- helper --
    private static boolean isUnaccessedState(double[] Pi, int index1) {
        return ( Pi[index1] == 0 && Pi[index1+1] == 0 && Pi[index1+2] == 0 && Pi[index1+3] == 0 );
    }

    // -- helper --
    // For state that the agent has accessed,
    // Average the prob if there are multiple directions taken in different trajectories
    private static void standardizeAccessedStates(double[] Pi, int j) {
        // Count directions performed in this same state
        int nDirecTaken = 0;
        for (int i = 0; i < 4; i++) { if (Pi[j+i] == 1) { nDirecTaken += 1; } }

        // Put averaged prob to performed directions, prob = 0 otherwise
        double averagedProb = 1.0 / (double)nDirecTaken;
        for (int i = 0; i < 4; i++) {
            if (Pi[j+i] == 1) { Pi[j+i] = averagedProb; } else { Pi[j+i] = 0; }
        }
    }


    /** ----------------------------------------- Transition Probability ------------------------------------------- **/

    /* getTransitionProbs
        This function generates a transition probability matrix(3d) in following format:
            1st layer: Action (n, e, s, w)
            2nd layer: Start state
            3rd layer: End state
    */
    public RealMatrix[] getTransitionProbs() {

        System.out.format("Getting transition probabilities\n\n");

        double[][][] TranP = new double[4][maxX*maxY][maxX*maxY];

        // Inner square
        for (int r = 0; r < maxX*maxY; r++) {
            int startx = r / maxY; int starty = r % maxY;
            if (starty < maxY-1) { TranP[0][r][r + 1]    = 1.0; } else { TranP[0][r][r] = 1.0; }
            if (startx < maxX-1) { TranP[1][r][r + maxY] = 1.0; } else { TranP[1][r][r] = 1.0; }
            if (starty > 0)      { TranP[2][r][r - 1]    = 1.0; } else { TranP[2][r][r] = 1.0; }
            if (startx > 0)      { TranP[3][r][r - maxY] = 1.0; } else { TranP[3][r][r] = 1.0; }
        }

        // Print out all transition probs for all 4 directions                                                  // TODO
//        for (int c = 0; c < 4; c++) {
//            System.out.format("DIRECTION: %d\n\n", c);
//
//            for (int i = 0; i < maxX * maxY; i++) {
//                for (int j = 0; j < maxX * maxY; j++) {
//                    System.out.format("%.2f ", TranP[c][i][j]);
//                }
//                System.out.format("\n");
//            }
//            System.out.format("\n");
//        }


        // Stored as Apache matrix
        RealMatrix[] TranP_M = new RealMatrix[4];
        for (int i = 0; i < 4; i++) { TranP_M[i] = MatrixUtils.createRealMatrix(TranP[i]); }

        return TranP_M;
    }


    /** -------------------------------------------- Composite Data ------------------------------------------------ **/

    /* getPpi
        Returns a maxX*maxY by maxX*maxY prob matrix,
        each entrance = sigma_a [ Pi(s, a) * TransP(s1, a, s2) ]
     */
    public static RealMatrix getPpi(RealVector Pi, RealMatrix[] TransP) {

        System.out.format("Getting Ppi\n\n");

        double[][] Ppi = new double[maxX*maxY][maxX*maxY];

        for (int r = 0; r < maxX*maxY; r++) {
            for (int c = 0; c < maxX*maxY; c++) {
                for (int i = 0; i < 4; i++) { Ppi[r][c] += Pi.getEntry((r*4+i)) * TransP[i].getEntry(r,c); }
            }
        }

//        for (int i = 0; i < maxX*maxY; i++) {
//            for (int j = 0; j < maxX*maxY; j++) {
//                System.out.format("%.2f ", Ppi[i][j]);
//            }
//            System.out.format("\n");
//        }
//        System.out.format("\n");


        return MatrixUtils.createRealMatrix(Ppi);

    }


    /* -------------------------------------------------------------------------------------------------------------- */
    /*                                                     MAIN                                                       */
    /* -------------------------------------------------------------------------------------------------------------- */

    public static void main(String[] args) throws IOException {

        episodeParser parser = new episodeParser(3, 5, 5, "demos/4demo0.episode");

//        GridLocation[] locations = parseEnvironment();

////        int[] initLoc = getStartPosition();
////        System.out.format("%d, %d", initLoc[0], initLoc[1]);
//
////        int[][] T = parseTrajectory();
////        for(int i = 0; i < T.length; i++) {
////            System.out.format("%d, %d, %d\n", T[i][0], T[i][1], T[i][2]);
////        }
//
////        double[] Pi = getPolicyFromTrajectory();
////        for (int i = 0; i < Pi.length; i++) {
////            if ((i+1) % 4 == 0) { printf("%f\n", Pi[i]); }
////            else { printf("%f ", Pi[i]); }
////        }
//
//        RealMatrix[] TransP = getTransitionProbs();
//        for (int i = 0; i < TransP[3].getData().length; i++) {
//            for (int j = 0; j < TransP[3].getRow(i).length; j++) {
//                printf("%f ", TransP[3].getEntry(i,j));
//            }
//            System.out.format("\n");
//        }
//
    }
}

