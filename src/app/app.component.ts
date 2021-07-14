import { Component, OnInit } from '@angular/core';
import { Participant } from './participant';
import { ParticipantService } from './participant.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})

export class AppComponent implements OnInit {
  participants: Participant[] = []
  classes = ["INTERACTIVE", "ATTENTIVE", "INATTENTIVE", "DROWSY"]
  constructor(private pservice: ParticipantService) { }

  ngOnInit(): void {
    this.getAllParticipants()
  }

  getAllParticipants() {
    this.pservice.getAllParticipants().subscribe(data => {
      this.participants = data
      this.ngOnInit()
    })
  }

  startApplication() {
    this.pservice.start().subscribe(data => {
      console.log(data);
    })
  }
}
