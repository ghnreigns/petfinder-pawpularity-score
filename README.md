1. This project is useful for Multi-Classification. Binary or Multi-Label need to tweak the shape of y_true and y_pred etc.
2. For BCEwithLogits loss, the shape of y_true and y_pred should be [batch_size, 1], so we might have to use .view(-1,1) to reshape it, see SETI.

---

25th November 2021: Adopting good commit and push messages.
