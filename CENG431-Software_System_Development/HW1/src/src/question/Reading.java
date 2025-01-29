package src.question;

import java.util.ArrayList;
import java.util.List;

import src.question.type.IType;
import src.question.type.Text;

public class Reading extends Question {
    private List<IType> questionTypes;

    public Reading() {
        this(10);
    }

    public Reading(int points) {
        super(points);
        createQuestionTypes();
    }

    private void createQuestionTypes() {
        questionTypes = new ArrayList<IType>();
        IType stringQuestion1 = new Text();
        IType stringQuestion2 = new Text();
        questionTypes.add(stringQuestion1);
        questionTypes.add(stringQuestion2);

    }

    public List<IType> getQuestionTypes() {
        return questionTypes;
    }

    @Override
    public String toString() {
        return "R";
    }
}
