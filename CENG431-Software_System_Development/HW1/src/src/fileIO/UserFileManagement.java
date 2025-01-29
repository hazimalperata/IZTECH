package src.fileIO;

import java.util.ArrayList;
import java.util.List;

import src.user.User;

public class UserFileManagement {
    private UserFileIO userFileIO;
    private List<String[]> userInfoList;

    public UserFileManagement() {
        userFileIO = new UserFileIO();
        userInfoList = new ArrayList<>();
    }

    public List<String[]> getUserInfoFromFile() {
        List<String> lineList = userFileIO.readFile();

        for (String line : lineList) {
            String[] values = line.split(";");
            String name = values[0];
            String password = values[1];
            userInfoList.add(new String[]{name, password});
        }
        return userInfoList;
    }

    public void writeUsers(List<User> users) {
        List<String> usersRow = new ArrayList<>();
        for (User user : users) {
            String row = user.getName() + ";" + user.getPassword() + ";" +
                    user.getChosenLanguage().getName() + ";" + user.getCurrentUnit() + ";" +
                    user.getNumberOfSolvedQuizzes() + ";" + user.getTotalPoints();
            usersRow.add(row);
        }
        userFileIO.writeFile(usersRow);
    }
}
