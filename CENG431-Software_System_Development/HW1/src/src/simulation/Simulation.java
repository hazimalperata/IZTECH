package src.simulation;


import src.fileIO.LanguagesFileManagement;
import src.fileIO.UserFileManagement;
import src.language.Language;
import src.league.Bronze;
import src.league.Gold;
import src.league.ILeague;
import src.league.League;
import src.league.Ruby;
import src.league.Sapphire;
import src.league.Silver;
import src.user.User;
import utils.Rand;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Simulation {
    UserFileManagement userFileManagement;
    List<String> languageNames;
    List<String[]> usersInfoList;
    List<User> users;

    LanguagesFileManagement languagesFileManagement;
    List<Language> languages;

    public Simulation() {
        userFileManagement = new UserFileManagement();
        languagesFileManagement = new LanguagesFileManagement();
        languages = new ArrayList<>();
        usersInfoList = new ArrayList<>();
        users = new ArrayList<>();
    }

    public void startSimulation(){
        createLanguages();
        createUsers();
        takeQuizzes();
        saveUsers();
        sortLeagues();
        User maxUser = getUserWhoHasMaxPoint();
        System.out.println("1- " + maxUser.getName() + " " + maxUser.getTotalPoints() + " points");
        User advancedUser = getUserWhoIsMostAdvancedInGerman();
        System.out.println("2- " + advancedUser.getName() + " " + advancedUser.getCurrentUnit().getName());
        Language maxUnitLanguage = getLanguageWhichHasMaxUnit();
        System.out.println("3- " + maxUnitLanguage.getName() + " " + maxUnitLanguage.getNumberOfUnits() + " Units");
        Language maxQuizzesLanguage = getLanguageWhichHasMaxQuizzes();
        System.out.println("4- " + maxQuizzesLanguage.getName() + " " + maxQuizzesLanguage.getNumberOfTotalQuizzes() + " Quizzes" );
        List<User> topThreeUsers = getTopThreeUserInSilverLeagueForItalian();
        System.out.printf("5- Italian Silver League Top 3: ");
        if (topThreeUsers.size() != 0) {
        	int index = 0;
        	for (User user : topThreeUsers) {
            	System.out.printf("%s.%s ",index+1,user.getName());
            	index += 1;
            }
        } else {
        	System.out.printf("Empty");
        }
    }
    private List<ILeague> getLeagues() {
        ILeague bronze = new Bronze("Bronze");
        ILeague silver = new Silver("Silver");
        ILeague gold = new Gold("Gold");
        ILeague sapphire = new Sapphire("Sapphire");
        ILeague ruby = new Ruby("Ruby");
        return new ArrayList<>(Arrays.asList(bronze, silver, gold, sapphire, ruby));
    }

    //Language dosyası yoksa
    private void createLanguages() {
        languageNames = Arrays.asList("Turkish", "German", "Italian", "Spanish");
        if (!languagesFileManagement.isExistFile()) {
            for (String language : languageNames) {
                Language languageObject = new Language(language, getLeagues());
                languages.add(languageObject);
            }
            languagesFileManagement.writeLanguages(languages);
        } else {
        	List<List<ILeague>> leagues = new ArrayList();
        	for (int i = 0; i < languageNames.size(); i++) {
        		leagues.add(getLeagues());
        	}
            languages = languagesFileManagement.getLanguagesListFromFile(leagues);
        }
    }

    private void createUsers() {
        usersInfoList = userFileManagement.getUserInfoFromFile();
        Rand<Language> randomListObject = new Rand<>();
        for (String[] userInfo : usersInfoList) {
            Language selectedLanguage = randomListObject.getListObject(languages);
            User newUser = new User(userInfo[0], userInfo[1], selectedLanguage);
            users.add(newUser);
        }
    }

    private void takeQuizzes() {
        for (User user : users) {
            user.solveQuizzes();
        }
    }

    private void sortLeagues() {
        for (Language language : this.languages) {
            language.handleLeagues();
        }
    }

    // TODO chosenLanguage, currentUnit, numberOfSolvedQuizzes, totalPoints kayit edilmeli/dosya olusturulmali. Daha sonra ise lig islemlerine gecilmeli. Liglerin yapisi konusulmali.

    private void saveUsers(){
        userFileManagement.writeUsers(users);
    }

    private User getUserWhoHasMaxPoint(){
        User maxUser = users.get(0);
        for (User user: users){
            if (maxUser.getTotalPoints() < user.getTotalPoints()){
                maxUser = user;
            }
        }
        return maxUser;
    }

    private User getUserWhoIsMostAdvancedInGerman(){
        List<User> germanUsers = new ArrayList<>();
        for (User user : users) {
            if (user.getChosenLanguage().getName().equals("German")){
                germanUsers.add(user);
            }
        }
        User advancedUser = germanUsers.get(0);
        for (User user: germanUsers){
            Integer advancedUserNumUnit = Integer.parseInt(advancedUser.getCurrentUnit().getName().substring(4).strip());
            Integer userNumUnit = Integer.parseInt(user.getCurrentUnit().getName().substring(4).strip());
            if (advancedUserNumUnit <  userNumUnit){
                advancedUser = user;
            }
        }
        return advancedUser;
    }

    private Language getLanguageWhichHasMaxUnit(){
        Language maxLanguage = languages.get(0);
        for (int i = 1; i < languages.size();  i++){
            if (maxLanguage.getNumberOfUnits() < languages.get(i).getNumberOfUnits()){
                maxLanguage = languages.get(i);
            }
        }
        return maxLanguage;
    }


    private Language getLanguageWhichHasMaxQuizzes(){
        Language maxLanguage = languages.get(0);
        for (int i = 1; i < languages.size(); i++){
            if (maxLanguage.getNumberOfTotalQuizzes() < languages.get(i).getNumberOfTotalQuizzes()){
                maxLanguage = languages.get(i);
            }
        }
        return maxLanguage;
    }

    private List<User> getTopThreeUserInSilverLeagueForItalian(){
        List<User> italianUsers = new ArrayList<>();
        for (Language language : languages) {
            if (language.getName().equals("Italian")){
                ILeague silverLeague = language.getSilverLeague();
                italianUsers.addAll(silverLeague.getTopUsers(3));
            }
        }

        return italianUsers;
    }
}
