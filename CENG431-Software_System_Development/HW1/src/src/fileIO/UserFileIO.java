package src.fileIO;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class UserFileIO {
    private String filePath = "users.csv";

    //default constructor
    public UserFileIO() {
    }

    public List<String> readFile() {
        String row;
        List<String> lines = new ArrayList<String>();
        BufferedReader csvReader = null;
        try {
            csvReader = new BufferedReader(new FileReader(filePath));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
        while (true) {
            try {
                if ((row = csvReader.readLine()) == null) break;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
            lines.add(row);
        }
        return new ArrayList<String>(lines);
    }

    public void writeFile(List<String> usersRow){
        try {
            FileWriter myWriter = new FileWriter(filePath);
            for (int i = 0; i < usersRow.size(); i++) {
                if (i == usersRow.size()-1){
                    myWriter.write(usersRow.get(i));
                }else{
                    myWriter.write(usersRow.get(i)+"\n");
                }

            }
            myWriter.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }
}
