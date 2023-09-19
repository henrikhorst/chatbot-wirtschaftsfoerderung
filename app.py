from dotenv import load_dotenv

# Load .env file
load_dotenv()

import streamlit as st
import pickle
import faiss
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

BASE_URL = "https://luebeck.org/"

template = """Angesichts der folgenden extrahierten Teile eines langen Dokuments und einer Frage, erstellen Sie eine endgültige Antwort mit Verweisen ("QUELLEN").
Wenn Sie die Antwort nicht kennen, sagen Sie einfach, dass Sie es nicht wissen. Versuchen Sie nicht, eine Antwort zu erfinden.
Geben Sie IMMER einen "QUELLEN"-Teil in Ihrer Antwort zurück.

FRAGE: Welchem Staat/Welchem Landesrecht unterliegt die Auslegung des Vertrages?
Inhalt: Dieses Abkommen unterliegt dem englischen Recht und die Parteien unterwerfen sich der ausschließlichen Zuständigkeit der englischen Gerichte in Bezug auf Streitigkeiten (vertraglich oder außervertraglich) bezüglich dieses Abkommens. Allerdings kann jede Partei bei einem Gericht einen Antrag auf eine einstweilige Verfügung oder eine andere Maßnahme stellen, um ihre geistigen Eigentumsrechte zu schützen.
Quelle: 28-pl
Inhalt: Kein Verzicht. Ein Versäumnis oder eine Verzögerung bei der Ausübung eines Rechts oder eines Rechtsmittels aus diesem Abkommen gilt nicht als Verzicht auf dieses (oder ein anderes) Recht oder Rechtsmittel.\n\n11.7 Trennbarkeit. Die Ungültigkeit, Rechtswidrigkeit oder Nichtdurchsetzbarkeit einer Bestimmung (oder eines Teils einer Bestimmung) dieses Abkommens beeinträchtigt nicht das Weiterbestehen des Rests der Bestimmung (falls vorhanden) und dieses Abkommens.\n\n11.8 Keine Agentur. Sofern nicht ausdrücklich anders angegeben, begründet nichts in diesem Abkommen eine Agentur, Partnerschaft oder Joint Venture jeglicher Art zwischen den Parteien.\n\n11.9 Keine Drittbegünstigten.
Quelle: 30-pl
Inhalt: (b) wenn Google in gutem Glauben der Ansicht ist, dass der Vertriebspartner Google dazu veranlasst hat, gegen Anti-Bestechungsgesetze (wie in Klausel 8.5 definiert) zu verstoßen, oder wenn ein solcher Verstoß wahrscheinlich eintreten wird,
Quelle: 4-pl
ENDANTWORT: Dieses Abkommen unterliegt dem englischen Recht.
QUELLEN: 28-pl

FRAGE: Was hat der Präsident über Michael Jackson gesagt?
Inhalt: Frau Sprecherin, Frau Vizepräsidentin, unsere First Lady und Second Gentleman. Mitglieder des Kongresses und des Kabinetts. Richter des Obersten Gerichtshofs. Meine amerikanischen Mitbürger. \n\nLetztes Jahr hat uns COVID-19 getrennt. Dieses Jahr sind wir endlich wieder zusammen. \n\nHeute Abend treffen wir uns als Demokraten, Republikaner und Unabhängige. Aber am wichtigsten als Amerikaner. \n\nMit einer Pflicht zueinander, zum amerikanischen Volk, zur Verfassung. \n\nUnd mit der unerschütterlichen Entschlossenheit, dass die Freiheit immer über die Tyrannei triumphieren wird. \n\nVor sechs Tagen versuchte Russlands Wladimir Putin, die Grundlagen der freien Welt zu erschüttern, in der Annahme, er könne sie seinen bedrohlichen Wegen unterwerfen. Aber er hat sich schwer verschätzt. \n\nEr dachte, er könnte in die Ukraine einmarschieren und die Welt würde sich fügen. Stattdessen stieß er auf eine Mauer der Stärke, die er sich nie vorgestellt hatte. \n\nEr traf auf das ukrainische Volk. \n\nVom Präsidenten Selenskyj bis zu jedem Ukrainer, ihre Furchtlosigkeit, ihr Mut, ihre Entschlossenheit inspiriert die Welt. \n\nGruppen von Bürgern, die Panzer mit ihren Körpern blockieren. Jeder von Studenten bis zu Rentnern, Lehrern, die zu Soldaten wurden und ihr Heimatland verteidigten.
Quelle: 0-pl
Inhalt: Und wir werden nicht aufhören. \n\nWir haben so viel durch COVID-19 verloren. Zeit miteinander. Und am schlimmsten, so viel Verlust von Leben. \n\nNutzen wir diesen Moment zum Neustart. Sehen wir COVID-19 nicht mehr als parteipolitische Trennlinie, sondern für das, was es ist: Eine schreckliche Krankheit. \n\nLassen Sie uns aufhören, einander als Feinde zu sehen und beginnen, einander so zu sehen, wie wir wirklich sind: Amerikanische Mitbürger. \n\nWir können nicht ändern, wie geteilt wir waren. Aber wir können ändern, wie wir vorankommen - bei COVID-19 und bei anderen Fragen, denen wir uns gemeinsam stellen müssen. \n\nIch besuchte vor kurzem das New Yorker Polizeidepartement wenige Tage nach den Beerdigungen von Officer Wilbert Mora und seinem Partner, Officer Jason Rivera. \n\nSie reagierten auf einen 9-1-1-Anruf, als ein Mann sie mit einer gestohlenen Waffe erschoss. \n\nOfficer Mora war 27 Jahre alt. \n\nOfficer Rivera war 22. \n\nBeide Dominikaner, die auf denselben Straßen aufwuchsen, die sie später als Polizisten patrouillierten. \n\nIch sprach mit ihren Familien und sagte ihnen, dass wir für immer in ihrer Schuld stehen für ihr Opfer, und wir werden ihre Mission fortsetzen, das Vertrauen und die Sicherheit wiederherzustellen, die jede Gemeinschaft verdient.
Quelle: 24-pl
Inhalt: Und ein stolzes ukrainisches Volk, das 30 Jahre Unabhängigkeit gekannt hat, hat wiederholt gezeigt, dass es niemanden tolerieren wird, der versucht, ihr Land zurückzudrängen. \n\nAn alle Amerikaner werde ich ehrlich zu Ihnen sein, wie ich es immer versprochen habe. Ein russischer Diktator, der ein fremdes Land überfällt, hat weltweite Kosten. \n\nUnd ich ergreife robuste Maßnahmen, um sicherzustellen, dass der Schmerz unserer Sanktionen sich gegen die russische Wirtschaft richtet. Und ich werde jedes uns zur Verfügung stehende Mittel nutzen, um amerikanische Unternehmen und Verbraucher zu schützen. \n\nHeute Abend kann ich verkünden, dass die Vereinigten Staaten mit 30 anderen Ländern zusammengearbeitet haben, um 60 Millionen Barrel Öl aus den Reserven weltweit freizugeben. \n\nAmerika wird diese Anstrengung anführen und 30 Millionen Barrel aus unserer eigenen strategischen Erdölreserve freigeben. Und wir sind bereit, bei Bedarf noch mehr zu tun, vereint mit unseren Verbündeten. \n\nDiese Schritte werden dazu beitragen, die Benzinpreise hier zu Hause abzufedern. Und ich weiß, dass die Nachrichten über das, was passiert, beunruhigend sein können. \n\nAber ich möchte, dass Sie wissen, dass es uns gut gehen wird.
Quelle: 5-pl
Inhalt: Mehr Unterstützung für Patienten und Familien. \n\nUm dorthin zu gelangen, fordere ich den Kongress auf, ARPA-H, die Advanced Research Projects Agency für Gesundheit, zu finanzieren. \n\nSie basiert auf DARPA - dem Verteidigungsprojekt, das zum Internet, GPS und vielem mehr geführt hat. \n\nARPA-H wird einen einzigen Zweck haben - Durchbrüche bei Krebs, Alzheimer, Diabetes und mehr zu fördern. \n\nEine Einheitsagenda für die Nation. \n\nWir können das schaffen. \n\nMeine amerikanischen Mitbürger - heute Abend haben wir uns an einem heiligen Ort versammelt - der Zitadelle unserer Demokratie. \n\nIn diesem Kapitol haben Generation um Generation Amerikaner große Fragen inmitten großer Konflikte diskutiert und Großes geleistet. \n\nWir haben für die Freiheit gekämpft, die Freiheit erweitert, Totalitarismus und Terror besiegt. \n\nUnd die stärkste, freieste und wohlhabendste Nation aufgebaut, die die Welt je gekannt hat. \n\nJetzt ist die Stunde. \n\nUnser Moment der Verantwortung. \n\nUnser Test von Entschlossenheit und Gewissen, der Geschichte selbst. \n\nIn diesem Moment wird unser Charakter geformt. Unser Zweck gefunden. Unsere Zukunft geschmiedet. \n\nNun, ich kenne diese Nation.
Quelle: 34-pl
ENDANTWORT: Der Präsident hat Michael Jackson nicht erwähnt.
QUELLEN:

FRAGE: {question}
{summaries}
ENDANTWORT:"""

llm=ChatOpenAI(temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])

with open("faiss_store_openai.pkl", "rb") as f:
    VectorStore = pickle.load(f)

GERMAN_QA_PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])
GERMAN_DOC_PROMPT = PromptTemplate(
    template="Inhalt: {page_content}\nQuelle: {source}",
    input_variables=["page_content", "source"])

qa_chain = load_qa_with_sources_chain(llm, chain_type="stuff",
                                      prompt=GERMAN_QA_PROMPT,
                                      document_prompt=GERMAN_DOC_PROMPT) 
chain = RetrievalQAWithSourcesChain(combine_documents_chain=qa_chain, retriever=VectorStore.as_retriever(),
                                     reduce_k_below_max_tokens=True, max_tokens_limit=3375,
                                     return_source_documents=True)

st.set_page_config(page_title="🤗💬 Wirtschaftsförderung Lübeck Chat")

with st.sidebar:
    st.title('🤗 Wirtschaftsförderung Lübeck')
    st.markdown('📖 Erfahren Sie mehr: [Zur Website](https://luebeck.org/)')

st.title('🎈 Chatbot')

st.write('Demo')


#Session state
#Initialize the chatbot by with messages session state and giving it a starter message at the first app run:
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Wie kann ich Ihnen helfen?"}]

#Here, past denotes the human user's input and generated indicates the bot's response.


#Display chat messages
#Conversational messages are displayed iteratively from the messages session state via the for loop together with the use of Streamlit’s chat feature st.chat_message().
# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


#Accept user prompt
#User’s input prompt message are accepted via the st.chat_input() method and appended to the messages session state followed by displaying the message via st.chat_message() together with st.write():
# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

#Generate bot response output
#Use an if condition to detect whether the last response in the messages session state is from the assistant or user. The chatbot will be triggered to generate a response if the last message is not from the chatbot (assistant). In generating the response, the st.chat_message(), st.spinner() and the custom generate_response() function are used where generated messages will display a spinner with a short message saying Thinking.... Finally, the generated response is saved to the messages session state.
# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Denke..."):
            response = chain({"question": prompt}, return_only_outputs=True)
            answer = response['answer'].replace('./html/', BASE_URL).replace(" QUELLEN: ", "\nQUELLEN: ")
            st.write(answer)
            print(answer)
    message = {"role": "assistant", "content": answer}
    st.session_state.messages.append(message)