package JAVA.Tasks.PasswordGenerator;
import java.util.Random;

public class PasswordGenerator {
    public static void main(String[] args) {
        int passwordLength = 10;
        System.out.println(generatePassword(passwordLength));
    }

    public static String generatePassword(int length) {
        String upperCaseLetters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
        String lowerCaseLetters = upperCaseLetters.toLowerCase();
        String digits = "0123456789";
        String all = upperCaseLetters + lowerCaseLetters + digits;

        Random random = new Random();
        StringBuilder password = new StringBuilder();

        for (int i = 0; i < length; i++) {
            int index = random.nextInt(all.length());
            password.append(all.charAt(index));
        }

        return password.toString();
    }
}
