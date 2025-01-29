package src.fileIO;
import java.io.*;
import java.util.ArrayList;
import java.util.List;

public class LanguagesFileIO {
    private final String filePath = "languages.csv";
    private boolean isExistFile = false;

    public LanguagesFileIO(){
        setExistFile();
    }
    private void setExistFile(){
        File file = new File(filePath);
        if (file.exists())
            isExistFile = true;
    }

    public boolean isExistFile() {
        return isExistFile;
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

    private void createFile(){
        if (!isExistFile) {
            try {

                File file = new File(filePath);
                if (file.createNewFile()) {
                    return;
                } else {
                    System.out.println("File already exists.");
                }
            } catch (IOException e) {
                System.out.println("An error occurred.");
                e.printStackTrace();
            }
        }
    }

    public void writeFile(List<String> languagesRow){
        createFile();
        try {
            FileWriter myWriter = new FileWriter(filePath);
            for (int i = 0; i < languagesRow.size(); i++) {
                if (i == languagesRow.size()-1){
                    myWriter.write(languagesRow.get(i));
                }else{
                    myWriter.write(languagesRow.get(i)+"\n");
                }

            }
            myWriter.close();
        } catch (IOException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
    }


}
