// document.addEventListener("DOMContentLoaded", function () {
//     // Votre logique d'initialisation va ici
// });

// function toggleChatBox() {
//     var chatBox = document.getElementById("chat-box");
//     chatBox.classList.toggle("hidden");
// }

// function sendMessage() {
//     var userInput = document.getElementById("user-input");
//     var message = userInput.value.trim();

//     if (message !== "") {
//         appendMessage("user", message);
//         userInput.value = "";
//         processMessage(message);
//     }
// }

// function appendMessage(sender, text) {
//     var chatMessages = document.getElementById("chat-messages");
//     var messageElement = document.createElement("div");
//     messageElement.className = sender;
//     messageElement.textContent = text;
//     chatMessages.appendChild(messageElement);

//     // Faites défiler vers le bas du conteneur de chat
//     chatMessages.scrollTop = chatMessages.scrollHeight;
// }

// function processMessage(message) {
//     // Votre logique de chatbot va ici
//     var response = getBotResponse(message);
//     appendMessage("bot", response);
// }

// function getBotResponse(userInput) {
//     // Ajoutez d'autres conditions spécifiques si nécessaire
//     var lowerCaseInput = userInput.toLowerCase();

//     if (lowerCaseInput.includes('coca-cola') || lowerCaseInput.includes('coca cola')) {
//         // Si l'utilisateur mentionne Coca-Cola, fournissez une réponse générale
//         return "Coca-Cola est une célèbre entreprise de boissons.";
//     } else if (lowerCaseInput.includes('stock') && lowerCaseInput.includes('data')) {
//         // Si l'utilisateur demande des données sur les actions, appelez la fonction pour obtenir les données sur les actions
//         var stockData = getStockData();
//         return "Voici les dernières données boursières de Coca-Cola : " + stockData;
//     } else if (lowerCaseInput.includes('fondateur') || lowerCaseInput.includes('créateur')) {
//         return "Coca-Cola a été créé par John Stith Pemberton.";
//     } else if (lowerCaseInput.includes('siege social') || lowerCaseInput.includes('emplacement')) {
//         return "Le siège social de Coca-Cola est à Atlanta, en Géorgie, États-Unis.";
//     } else {
//         // Réponse par défaut si aucune correspondance spécifique
//         return "Je suis un chatbot simple. Vous avez dit : '" + userInput + "'";
//     }
// }

// // Vous pouvez ajouter d'autres fonctions et logique selon vos besoins
