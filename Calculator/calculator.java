package JAVA.Tasks;
import java.util.*;
public class calculator {
    public static void main(String[] args){
        Scanner sc = new Scanner(System.in);
        System.out.println("Choose any one (1. Add, 2. Sub, 3. Multiply, 4. Divide) : ");
        int choice = sc.nextInt();
        System.out.println("Enter first number : ");
        float a = sc.nextFloat();
        System.out.println("Enter second number : ");
        float b = sc.nextFloat();
        if(choice == 1){
        float c = a+b;
        System.out.println("Sum of  "+a+" and "+b+" is : "+c);
        } else if (choice == 2) {
        float d = a-b;
        System.out.println("Difference of  "+a+" and "+b+" is : "+d);
        } else if (choice == 3) {
            float e = a*b;
            System.out.println("Product of  "+a+" and "+b+" is : "+e);
        } else if (choice == 4) {
            float f = a/b;
            System.out.println("Division of  "+a+" and "+b+" gives : "+f);
        }
        else {
            System.out.println("Invalid Input!!");
        }
        System.out.println("Thank You !!");
    }
}
