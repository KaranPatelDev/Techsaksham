package JAVA.Tasks;
import java.util.*;
public class number_guessing_game {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        Random rand = new Random();
        int number_guess = rand.nextInt(100) + 1;
        int numberoftrials = 0;
        int guess;
        boolean win = false;
        System.out.println("Guess a number between 1 and 100 : ");
        while (!win) {
            guess = sc.nextInt();
            numberoftrials++;
            if (guess == number_guess) {
                win = true;
            } else if (guess < number_guess) {
                System.out.println("Too low, try again:");
            } else if (guess > number_guess) {
                System.out.println("Too high, try again:");
            }

        }
            System.out.println("You win!");
            System.out.println("The number was " + number_guess);
            System.out.println("It took you " + numberoftrials + " tries.");

    }
}
